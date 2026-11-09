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
#include <yantra/index.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

auto
main (int argc, char **argv) -> int
{
  constexpr std::size_t kN = 1000000U;
  constexpr std::size_t kD = 128U;
  constexpr std::size_t kM = 16U;
  constexpr std::size_t kEfC = 200U;
  constexpr std::size_t kEfS = 64U;
  constexpr std::size_t kQueries = 100U;
  constexpr std::size_t kTopK = 10U;

  bool use_persisted = false;
  for (int i = 1; i < argc; ++i)
    {
      if (std::string (argv[i]) == "--use-persisted")
        {
          use_persisted = true;
        }
    }

  std::vector<float> vecs (kN * kD);
  const fs::path vec_path = "benchmarks/data/sift1m-like.bin";
  if (fs::exists (vec_path)
      && fs::file_size (vec_path) == kN * kD * sizeof (float))
    {
      std::ifstream in (vec_path, std::ios::binary);
      in.read (reinterpret_cast<char *> (vecs.data ()),
               static_cast<std::streamsize> (kN * kD * sizeof (float)));
    }
  else
    {
      std::mt19937_64 rng (0xDEADBEEFULL);
      std::normal_distribution<float> dist (0.0F, 1.0F);
      for (float &x : vecs)
        {
          x = dist (rng);
        }
    }

  std::vector<float> queries (kQueries * kD);
  const fs::path q_path = "benchmarks/data/queries-1k.bin";
  if (fs::exists (q_path)
      && fs::file_size (q_path) >= kQueries * kD * sizeof (float))
    {
      std::ifstream in (q_path, std::ios::binary);
      in.read (reinterpret_cast<char *> (queries.data ()),
               static_cast<std::streamsize> (kQueries * kD * sizeof (float)));
    }
  else
    {
      std::mt19937_64 qrng (0xDEADBEEFULL ^ 0x1000ULL);
      std::normal_distribution<float> dist (0.0F, 1.0F);
      for (float &x : queries)
        {
          x = dist (qrng);
        }
    }

  yantra::HnswIndex hnsw (kD, { kM, kEfC, kEfS });
  const fs::path persisted_path = "benchmarks/data/hnsw_1m.yhnsw";
  const uint64_t t_build0 = yantra::monotonic_time_ns ();
  if (use_persisted && fs::exists (persisted_path))
    {
      auto loaded = yantra::HnswIndex::load_mmap (persisted_path);
      assert (loaded.has_value ());
      hnsw = std::move (*loaded);
    }
  else
    {
      auto b = hnsw.add_batch (vecs, kN);
      assert (b.has_value ());
    }
  const double build_s
      = static_cast<double> (yantra::monotonic_time_ns () - t_build0) / 1.0e9;

  /* B1. Layer distribution  */
  const std::size_t max_lvl = hnsw.max_level ();
  std::vector<std::size_t> level_counts (max_lvl + 1U, 0U);
  for (std::size_t u = 0U; u < kN; ++u)
    {
      const std::size_t l = hnsw.node_level (u);
      for (std::size_t k = 0U; k <= l && k <= max_lvl; ++k)
        {
          ++level_counts[k];
        }
    }
  const double level1_frac
      = (max_lvl >= 1U)
            ? (static_cast<double> (level_counts[1])
               / static_cast<double> (kN))
            : 0.0;

  /* B2. Layer-0 degree histogram & reciprocity  */
  std::vector<std::size_t> degrees (kN);
  std::size_t low_deg_count = 0U;
  std::size_t total_directed_edges = 0U;
  std::size_t reciprocal_edges = 0U;

  for (std::size_t u = 0U; u < kN; ++u)
    {
      auto nbrs = hnsw.layer0_neighbors (u);
      degrees[u] = nbrs.size ();
      if (degrees[u] < (kM / 4U))
        {
          ++low_deg_count;
        }
      total_directed_edges += nbrs.size ();
      for (uint32_t v : nbrs)
        {
          if (v < kN)
            {
              auto v_nbrs = hnsw.layer0_neighbors (v);
              for (uint32_t w : v_nbrs)
                {
                  if (w == u)
                    {
                      ++reciprocal_edges;
                      break;
                    }
                }
            }
        }
    }

  std::vector<std::size_t> sorted_deg = degrees;
  std::sort (sorted_deg.begin (), sorted_deg.end ());
  const std::size_t deg_min = sorted_deg.front ();
  const std::size_t deg_med = sorted_deg[kN / 2U];
  const std::size_t deg_max = sorted_deg.back ();
  const double low_deg_frac
      = static_cast<double> (low_deg_count) / static_cast<double> (kN);
  const double reciprocity_frac
      = total_directed_edges > 0U
            ? static_cast<double> (reciprocal_edges)
                  / static_cast<double> (total_directed_edges)
            : 0.0;

  /* B3. Directed BFS connectivity from a random level-0 node  */
  std::mt19937_64 bfs_rng (0xC0FFEEULL);
  const uint32_t start_node
      = static_cast<uint32_t> (bfs_rng () % static_cast<uint64_t> (kN));
  std::vector<uint8_t> visited (kN, 0U);
  std::queue<uint32_t> q;
  visited[start_node] = 1U;
  q.push (start_node);
  std::size_t reachable = 0U;

  while (!q.empty ())
    {
      const uint32_t u = q.front ();
      q.pop ();
      ++reachable;
      auto nbrs = hnsw.layer0_neighbors (u);
      for (uint32_t v : nbrs)
        {
          if (v < kN && visited[v] == 0U)
            {
              visited[v] = 1U;
              q.push (v);
            }
        }
    }
  const double reachable_frac
      = static_cast<double> (reachable) / static_cast<double> (kN);

  /* C. Run 100 queries and compute Recall@10 vs exact Flat search  */
  std::vector<uint64_t> gt_ids (kQueries * kTopK);
  const fs::path gt_path = "benchmarks/data/gt-1k-top10.bin";
  if (fs::exists (gt_path)
      && fs::file_size (gt_path) >= kQueries * kTopK * sizeof (uint64_t))
    {
      std::ifstream in (gt_path, std::ios::binary);
      in.read (
          reinterpret_cast<char *> (gt_ids.data ()),
          static_cast<std::streamsize> (kQueries * kTopK * sizeof (uint64_t)));
    }
  else
    {
      yantra::FlatIndex flat (kD);
      (void)flat.add_batch (vecs, kN);
      for (std::size_t qi = 0U; qi < kQueries; ++qi)
        {
          std::span<const float> qv (queries.data () + qi * kD, kD);
          auto g = flat.search (qv, kTopK);
          assert (g.has_value ());
          for (std::size_t k = 0U; k < kTopK; ++k)
            {
              gt_ids[qi * kTopK + k] = (*g)[k].id;
            }
        }
    }

  std::size_t total_hits = 0U;
  std::size_t min_hits = kTopK;
  std::vector<double> search_ms;
  search_ms.reserve (kQueries);

  for (std::size_t qi = 0U; qi < kQueries; ++qi)
    {
      std::span<const float> qv (queries.data () + qi * kD, kD);
      const uint64_t ts0 = yantra::monotonic_time_ns ();
      auto res = hnsw.search (qv, kTopK, kEfS);
      const uint64_t ts1 = yantra::monotonic_time_ns ();
      search_ms.push_back (static_cast<double> (ts1 - ts0) / 1.0e6);
      assert (res.has_value ());

      std::unordered_set<uint64_t> gt_set (
          gt_ids.begin () + static_cast<std::ptrdiff_t> (qi * kTopK),
          gt_ids.begin () + static_cast<std::ptrdiff_t> ((qi + 1U) * kTopK));
      std::size_t hits = 0U;
      for (const auto &r : *res)
        {
          if (gt_set.contains (r.id))
            {
              ++hits;
            }
        }
      total_hits += hits;
      min_hits = std::min (min_hits, hits);
    }

  std::sort (search_ms.begin (), search_ms.end ());
  const double p50_search_ms = search_ms[kQueries / 2U];
  const double mean_recall
      = static_cast<double> (total_hits)
        / static_cast<double> (kQueries * kTopK);
  const double worst_recall
      = static_cast<double> (min_hits) / static_cast<double> (kTopK);

  /* D. Dump JSON summary to diagnostics/hnsw_1m_diagnostic.json  */
  fs::create_directories ("diagnostics");
  {
    std::ofstream out ("diagnostics/hnsw_1m_diagnostic.json");
    out << std::fixed << std::setprecision (6);
    out << "{\n";
    out << "  \"num_nodes\": " << kN << ",\n";
    out << "  \"dim\": " << kD << ",\n";
    out << "  \"build_or_load_seconds\": " << build_s << ",\n";
    out << "  \"max_level\": " << max_lvl << ",\n";
    out << "  \"level_1_fraction\": " << level1_frac << ",\n";
    out << "  \"expected_level_1_fraction\": 0.062500,\n";
    out << "  \"layer_counts\": [";
    for (std::size_t l = 0U; l < level_counts.size (); ++l)
      {
        out << level_counts[l] << (l + 1U < level_counts.size () ? ", " : "");
      }
    out << "],\n";
    out << "  \"layer0_degree\": {\n";
    out << "    \"min\": " << deg_min << ",\n";
    out << "    \"median\": " << deg_med << ",\n";
    out << "    \"max\": " << deg_max << ",\n";
    out << "    \"fraction_below_m_div_4\": " << low_deg_frac << ",\n";
    out << "    \"reciprocity_fraction\": " << reciprocity_frac << "\n";
    out << "  },\n";
    out << "  \"connectivity\": {\n";
    out << "    \"start_node\": " << start_node << ",\n";
    out << "    \"reachable_nodes\": " << reachable << ",\n";
    out << "    \"reachable_fraction\": " << reachable_frac << "\n";
    out << "  },\n";
    out << "  \"search_100q\": {\n";
    out << "    \"p50_ms\": " << p50_search_ms << ",\n";
    out << "    \"mean_recall_at_10\": " << mean_recall << ",\n";
    out << "    \"worst_recall_at_10\": " << worst_recall << "\n";
    out << "  }\n";
    out << "}\n";
  }

  std::cout << "[diag_hnsw_1m] max_level=" << max_lvl
            << " level1_frac=" << level1_frac
            << " deg(min/med/max)=" << deg_min << "/" << deg_med << "/"
            << deg_max << " reciprocity=" << reciprocity_frac
            << " reachable_frac=" << reachable_frac
            << " mean_recall@10=" << mean_recall
            << " worst_recall@10=" << worst_recall
            << " p50_ms=" << p50_search_ms << "\n";

  const bool level_ok = (level1_frac >= 0.050 && level1_frac <= 0.075);
  const bool conn_ok = (reachable_frac >= 0.999);
  const bool recall_ok = (mean_recall >= 0.95 && worst_recall >= 0.80);

  if (!level_ok || !conn_ok || !recall_ok)
    {
      std::cerr << "[diag_hnsw_1m] FAILED: level_ok=" << level_ok
                << " conn_ok=" << conn_ok << " recall_ok=" << recall_ok
                << "\n";
      return 1;
    }
  return 0;
}
