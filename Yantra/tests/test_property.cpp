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

#include <algorithm>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace
{

/* Property 7.2.1: Replaying the WAL produces the exact same final state as the
   live ExperimentStore state for arbitrary metric log sequences.  */
void
prop_wal_replay_equivalence ()
{
  const fs::path dir = fs::temp_directory_path () / "yantra_prop_wal";
  std::error_code ec;
  fs::remove_all (dir, ec);

  std::mt19937 rng (1337U);
  std::uniform_real_distribution<double> val_dist (-50.0, 50.0);

  std::vector<yantra::RunRecord> live_snapshot;
  {
    auto store_res = yantra::ExperimentStore::open (dir);
    assert (store_res.has_value ());
    for (int r = 0; r < 20; ++r)
      {
        const std::string r_id = "prop_run_" + std::to_string (r);
        auto run_res = store_res->create_run (r_id);
        assert (run_res.has_value ());
        assert (run_res->log_param ("seed", std::to_string (r)).has_value ());
        for (uint64_t step = 0U; step < 30U; ++step)
          {
            assert (run_res->log_metric (step, "loss", val_dist (rng))
                        .has_value ());
          }
        assert (run_res->finish ().has_value ());
      }
    auto q_res = store_res->query ({});
    assert (q_res.has_value ());
    live_snapshot = *q_res;
  }

  /* Re-open store from disk to trigger WAL + metadata replay.  */
  auto replayed_store = yantra::ExperimentStore::open (dir);
  assert (replayed_store.has_value ());
  auto replay_q = replayed_store->query ({});
  assert (replay_q.has_value ());
  assert (replay_q->size () == live_snapshot.size ());

  for (std::size_t i = 0U; i < live_snapshot.size (); ++i)
    {
      const auto &a = live_snapshot[i];
      const auto &b = (*replay_q)[i];
      assert (a.run_id == b.run_id);
      assert (a.params.at ("seed") == b.params.at ("seed"));
      const auto &ma = a.metrics.at ("loss");
      const auto &mb = b.metrics.at ("loss");
      assert (ma.count == mb.count);
      assert (std::abs (ma.min_val - mb.min_val) < 1.0e-12);
      assert (std::abs (ma.max_val - mb.max_val) < 1.0e-12);
      assert (std::abs (ma.last_val - mb.last_val) < 1.0e-12);
    }

  fs::remove_all (dir, ec);
}

/* Property 7.2.2: Registry version numbers are monotonically increasing and no
   version is lost across concurrent registrations.  */
void
prop_registry_monotonic_concurrency ()
{
  const fs::path dir = fs::temp_directory_path () / "yantra_prop_reg";
  std::error_code ec;
  fs::remove_all (dir, ec);

  auto reg_res = yantra::ModelRegistry::open (dir);
  assert (reg_res.has_value ());
  auto &reg = *reg_res;

  constexpr int kThreads = 8;
  constexpr int kPerThread = 15;
  std::vector<std::jthread> workers;
  workers.reserve (kThreads);

  for (int t = 0; t < kThreads; ++t)
    {
      workers.emplace_back ([&, t] () {
        std::vector<std::byte> payload (64U,
                                        static_cast<std::byte> (t + 1));
        for (int i = 0; i < kPerThread; ++i)
          {
            const std::string tag
                = "v" + std::to_string (t) + "." + std::to_string (i);
            yantra::ModelLineage lin{};
            lin.run_id = tag;
            auto m = reg.register_version ("concurrent-net", tag, payload, lin,
                                           1U, false);
            assert (m.has_value ());
          }
      });
    }
  workers.clear ();

  auto versions = reg.list_versions ("concurrent-net");
  assert (versions.has_value ());
  assert (versions->size ()
          == static_cast<std::size_t> (kThreads * kPerThread));
  for (std::size_t i = 0U; i < versions->size (); ++i)
    {
      assert ((*versions)[i].version == i + 1U);
    }

  fs::remove_all (dir, ec);
}

/* Property 7.2.3: For any acyclic graph, DagScheduler produces a valid
   topological execution order; for any cyclic graph, it rejects with a typed
   DagError before executing any task.  */
void
prop_dag_topological_and_cycle_rejection ()
{
  yantra::DagScheduler sched (6U);
  std::mt19937 rng (2026U);

  for (int trial = 0; trial < 10; ++trial)
    {
      yantra::PipelineDefinition dag;
      constexpr int kNodes = 30;
      for (int i = 0; i < kNodes; ++i)
        {
          yantra::Task t{};
          t.id = "n_" + std::to_string (i);
          for (int p = 0; p < i; ++p)
            {
              if ((rng () % 5U) == 0U)
                {
                  t.dependencies.push_back ("n_" + std::to_string (p));
                }
            }
          t.run = [] (yantra::TaskContext &)
              -> yantra::Result<std::vector<std::string>> {
            return std::vector<std::string>{};
          };
          assert (dag.add_task (std::move (t)).has_value ());
        }

      auto rep = sched.execute (dag);
      assert (rep.has_value () && rep->all_succeeded);
      assert (rep->results.size () == static_cast<std::size_t> (kNodes));

      for (const auto &task : dag.tasks ())
        {
          const auto &child_res = rep->results.at (task.id);
          for (const auto &dep : task.dependencies)
            {
              const auto &parent_res = rep->results.at (dep);
              assert (child_res.start_timestamp_ns
                      >= parent_res.end_timestamp_ns);
            }
        }
    }

  /* Inject a cycle into a 10-node graph and verify zero tasks execute.  */
  std::atomic<int> executed_count{ 0 };
  yantra::PipelineDefinition cyclic_dag;
  for (int i = 0; i < 10; ++i)
    {
      yantra::Task t{};
      t.id = "c_" + std::to_string (i);
      t.dependencies.push_back ("c_" + std::to_string ((i + 1) % 10));
      t.run = [&] (yantra::TaskContext &)
          -> yantra::Result<std::vector<std::string>> {
        executed_count.fetch_add (1);
        return std::vector<std::string>{};
      };
      assert (cyclic_dag.add_task (std::move (t)).has_value ());
    }
  auto cyc_res = sched.execute (cyclic_dag);
  assert (!cyc_res.has_value () && cyc_res.error ().is_dag_error ());
  assert (executed_count.load () == 0);
}

/* Property 7.2.4: Vector index search results achieve >= 95% Recall@10 against
   exact brute-force ground truth.  */
void
prop_vector_index_recall_bound ()
{
  constexpr std::size_t kN = 2000U;
  constexpr std::size_t kD = 64U;
  std::mt19937 rng (99U);
  std::uniform_real_distribution<float> dist (-1.0F, 1.0F);

  std::vector<float> vecs (kN * kD);
  for (float &x : vecs)
    {
      x = dist (rng);
    }

  yantra::FlatIndex flat (kD);
  yantra::HnswIndex hnsw (kD, { 16U, 150U, 64U });
  assert (flat.add_batch (vecs, kN).has_value ());
  assert (hnsw.add_batch (vecs, kN).has_value ());

  std::size_t hits = 0U;
  constexpr std::size_t kQueries = 25U;
  constexpr std::size_t kTopK = 10U;

  for (std::size_t q = 0U; q < kQueries; ++q)
    {
      const std::span<const float> qvec (vecs.data () + (q * 37U % kN) * kD,
                                         kD);
      auto gt = flat.search_exact_fp64 (qvec, kTopK);
      auto ap = hnsw.search (qvec, kTopK, 64U);
      assert (gt.has_value () && ap.has_value ());
      std::unordered_set<uint64_t> gt_ids;
      for (const auto &r : *gt)
        {
          gt_ids.insert (r.id);
        }
      for (const auto &r : *ap)
        {
          if (gt_ids.contains (r.id))
            {
              ++hits;
            }
        }
    }

  const double recall
      = static_cast<double> (hits) / static_cast<double> (kQueries * kTopK);
  assert (recall >= 0.95);
}

} /* namespace */

auto
main () -> int
{
  prop_wal_replay_equivalence ();
  prop_registry_monotonic_concurrency ();
  prop_dag_topological_and_cycle_rejection ();
  prop_vector_index_recall_bound ();
  std::cout << "All Yantra property-based tests PASSED.\n";
  return 0;
}
