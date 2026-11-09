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

#include <cassert>
#include <cstdint>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

auto
main () -> int
{
  constexpr std::size_t kN = 100000U;
  constexpr std::size_t kD = 32U;
  constexpr std::size_t kM = 16U;

  std::mt19937_64 rng (0xDEADBEEFULL);
  std::normal_distribution<float> dist (0.0F, 1.0F);
  std::vector<float> vecs (kN * kD);
  for (float &x : vecs)
    {
      x = dist (rng);
    }

  yantra::HnswIndex hnsw (kD, { kM, 100U, 64U });
  auto build_res = hnsw.add_batch (vecs, kN);
  assert (build_res.has_value ());
  assert (hnsw.size () == kN);

  /* 1. Verify exponential layer assignment (level 1 ~ 6.25% +/- 20%).  */
  const std::size_t max_lvl = hnsw.max_level ();
  std::size_t level1_cnt = 0U;
  for (std::size_t u = 0U; u < kN; ++u)
    {
      if (hnsw.node_level (u) >= 1U)
        {
          ++level1_cnt;
        }
    }
  const double level1_frac
      = static_cast<double> (level1_cnt) / static_cast<double> (kN);
  std::cout << "[test_hnsw_connectivity] N=" << kN << " max_level=" << max_lvl
            << " level1_frac=" << level1_frac << "\n";
  assert (max_lvl >= 2U);
  assert (level1_frac >= 0.050 && level1_frac <= 0.075);

  /* 2. Verify Level 0 directed BFS connectivity >= 99.9% from start=99999.  */
  std::vector<uint8_t> visited (kN, 0U);
  std::queue<uint32_t> q;
  const uint32_t start_node = static_cast<uint32_t> (kN - 1U);
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

  const double conn_frac
      = static_cast<double> (reachable) / static_cast<double> (kN);
  std::cout << "[test_hnsw_connectivity] reachable=" << reachable << "/" << kN
            << " (" << conn_frac * 100.0 << "%)\n";
  assert (conn_frac >= 0.999);

  return 0;
}
