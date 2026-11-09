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

#include "yantra/index.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <span>
#include <unordered_set>
#include <vector>

auto
main (int argc, char **argv) -> int
{
  const bool run_1m = (argc > 1 && std::strcmp (argv[1], "--1m") == 0);
  constexpr std::size_t kNumVectors = 100000U;
  constexpr std::size_t kDim = 128U;
  constexpr std::size_t kTopK = 10U;
  constexpr std::size_t kNumQueries = 50U;

  std::vector<float> vectors (kNumVectors * kDim);
  std::mt19937_64 rng (20261101ULL);
  std::normal_distribution<float> dist (0.0F, 1.0F);
  for (float &v : vectors)
    {
      v = dist (rng);
    }

  yantra::FlatIndex flat (kDim);
  if (!flat.add_batch (vectors, kNumVectors).has_value ())
    {
      std::fprintf (stderr, "FlatIndex::add_batch failed\n");
      return 1;
    }

  yantra::IvfPqParams params;
  params.nlist = 256U;
  params.m_subspaces = 16U;
  params.nprobe = 16U;
  params.rerank_k = 64U;

  yantra::IvfPqIndex ivf (kDim, params);
  if (!ivf.build (vectors, kNumVectors).has_value ())
    {
      std::fprintf (stderr, "IvfPqIndex::build failed\n");
      return 1;
    }

  std::vector<float> queries (kNumQueries * kDim);
  for (float &q : queries)
    {
      q = dist (rng);
    }

  double recall_sum = 0.0;
  std::vector<double> latencies_ms;
  latencies_ms.reserve (kNumQueries);

  for (std::size_t q_i = 0U; q_i < kNumQueries; ++q_i)
    {
      const std::span<const float> q (queries.data () + q_i * kDim, kDim);
      auto gt_res = flat.search_exact_fp64 (q, kTopK);
      if (!gt_res.has_value ())
        {
          return 1;
        }
      std::unordered_set<uint64_t> gt_ids;
      for (const auto &r : *gt_res)
        {
          gt_ids.insert (r.id);
        }

      const auto t0 = std::chrono::steady_clock::now ();
      auto res = ivf.search (q, kTopK, 16U);
      const auto t1 = std::chrono::steady_clock::now ();
      if (!res.has_value ())
        {
          return 1;
        }
      const double ms
          = std::chrono::duration<double, std::milli> (t1 - t0).count ();
      latencies_ms.push_back (ms);

      std::size_t hits = 0U;
      for (const auto &r : *res)
        {
          if (gt_ids.contains (r.id))
            {
              ++hits;
            }
        }
      recall_sum
          += static_cast<double> (hits) / static_cast<double> (kTopK);
    }

  std::sort (latencies_ms.begin (), latencies_ms.end ());
  const double p50_ms = latencies_ms[kNumQueries / 2U];
  const double mean_recall = recall_sum / static_cast<double> (kNumQueries);

  std::fprintf (stderr,
                "[test_ivfpq_recall] N=%zu p50_ms=%.4f mean_recall@10=%.4f\n",
                kNumVectors, p50_ms, mean_recall);

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
  constexpr double kMaxP50Ms = 15.0;
#else
  constexpr double kMaxP50Ms = 0.50;
#endif
  if (mean_recall < 0.90 || p50_ms > kMaxP50Ms)
    {
      std::fprintf (stderr,
                    "[test_ivfpq_recall] FAILED (target recall>=0.90, "
                    "p50<=%.2fms)\n",
                    kMaxP50Ms);
      return 1;
    }

  if (run_1m)
    {
      constexpr std::size_t kN1M = 1000000U;
      constexpr std::size_t kQ1K = 1000U;
      std::vector<float> vecs_1m (kN1M * kDim);
      std::vector<float> q_1k (kQ1K * kDim);
      std::vector<uint64_t> gt_1k (kQ1K * kTopK);
      std::ifstream fv ("benchmarks/data/sift1m-like.bin", std::ios::binary);
      std::ifstream fq ("benchmarks/data/queries-1k.bin", std::ios::binary);
      std::ifstream fg ("benchmarks/data/gt-1k-top10.bin", std::ios::binary);
      if (fv && fq && fg)
        {
          fv.read (reinterpret_cast<char *> (vecs_1m.data ()),
                   static_cast<std::streamsize> (vecs_1m.size ()
                                                 * sizeof (float)));
          fq.read (
              reinterpret_cast<char *> (q_1k.data ()),
              static_cast<std::streamsize> (q_1k.size () * sizeof (float)));
          fg.read (reinterpret_cast<char *> (gt_1k.data ()),
                   static_cast<std::streamsize> (gt_1k.size ()
                                                 * sizeof (uint64_t)));

          yantra::IvfPqIndex ivf_1m (kDim, { 256U, 16U, 12U, 48U });
          const auto tb0 = std::chrono::steady_clock::now ();
          if (!ivf_1m.build (vecs_1m, kN1M).has_value ())
            {
              return 1;
            }
          const auto tb1 = std::chrono::steady_clock::now ();
          const double build_s
              = std::chrono::duration<double> (tb1 - tb0).count ();

          std::vector<double> lat_1m;
          lat_1m.reserve (kQ1K);
          std::size_t hits_1m = 0U;
          std::size_t min_hits = kTopK;
          for (std::size_t q = 0U; q < kQ1K; ++q)
            {
              const std::span<const float> qv (q_1k.data () + q * kDim, kDim);
              const auto ts0 = std::chrono::steady_clock::now ();
              auto r = ivf_1m.search (qv, kTopK, 12U);
              const auto ts1 = std::chrono::steady_clock::now ();
              lat_1m.push_back (
                  std::chrono::duration<double, std::milli> (ts1 - ts0)
                      .count ());
              std::unordered_set<uint64_t> gset (
                  gt_1k.begin () + static_cast<std::ptrdiff_t> (q * kTopK),
                  gt_1k.begin ()
                      + static_cast<std::ptrdiff_t> ((q + 1U) * kTopK));
              std::size_t qh = 0U;
              for (const auto &nb : *r)
                {
                  if (gset.contains (nb.id))
                    {
                      ++qh;
                    }
                }
              hits_1m += qh;
              min_hits = std::min (min_hits, qh);
            }
          std::sort (lat_1m.begin (), lat_1m.end ());
          const double p50_1m = lat_1m[kQ1K / 2U];
          const double p99_1m = lat_1m[(kQ1K * 99U) / 100U];
          const double rec_1m = static_cast<double> (hits_1m)
                                / static_cast<double> (kQ1K * kTopK);
          const double worst_1m
              = static_cast<double> (min_hits) / static_cast<double> (kTopK);
          std::fprintf (stderr,
                        "[test_ivfpq_recall 1M] build_s=%.4f p50_ms=%.4f "
                        "p99_ms=%.4f mean_recall@10=%.4f worst=%.4f\n",
                        build_s, p50_1m, p99_1m, rec_1m, worst_1m);
          if (rec_1m < 0.90 || p50_1m > 0.50 || build_s > 30.0)
            {
              return 1;
            }
        }
    }

  return 0;
}
