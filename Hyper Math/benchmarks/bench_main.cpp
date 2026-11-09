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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace hyper_math;
using namespace hyper_math::tensor;
using namespace hyper_math::scientific;
using namespace hyper_math::tabular;

namespace
{

double
read_rss_mb ()
{
  std::ifstream in ("/proc/self/status");
  std::string line;
  while (std::getline (in, line))
    {
      if (line.rfind ("VmRSS:", 0) == 0)
        {
          return static_cast<double> (std::stol (line.substr (6))) / 1024.0;
        }
    }
  return 0.0;
}

double
read_hwm_mb ()
{
  std::ifstream in ("/proc/self/status");
  std::string line;
  while (std::getline (in, line))
    {
      if (line.rfind ("VmHWM:", 0) == 0)
        {
          return static_cast<double> (std::stol (line.substr (6))) / 1024.0;
        }
    }
  return read_rss_mb ();
}

struct Stats
{
  double p50_ms = 0.0;
  double p99_ms = 0.0;
};

template <typename Fn>
Stats
measure_stats (Fn &&fn, int warmup = 4, int repeats = 15, int batch_iters = 4)
{
  for (int w = 0; w < warmup; ++w)
    {
      fn ();
    }
  std::vector<double> samples (static_cast<std::size_t> (repeats));
  for (int r = 0; r < repeats; ++r)
    {
      const auto t0 = std::chrono::steady_clock::now ();
      for (int b = 0; b < batch_iters; ++b)
        {
          fn ();
        }
      const auto t1 = std::chrono::steady_clock::now ();
      samples[static_cast<std::size_t> (r)] =
        std::chrono::duration<double, std::milli> (t1 - t0).count ()
        / static_cast<double> (batch_iters);
    }
  std::sort (samples.begin (), samples.end ());
  const std::size_t n = samples.size ();
  const std::size_t idx99 = std::min (
    n - 1,
    static_cast<std::size_t> (std::ceil (static_cast<double> (n) * 0.99) - 1.0));
  return Stats{samples[n / 2], samples[idx99]};
}

struct MetricEntry
{
  std::string name;
  double value = 0.0;
  std::string unit;
  std::string methodology_ref;
  bool sanity_passed = false;
  std::string sanity_details;
  bool higher_is_better = false;
  bool check_regression = true;
};

std::string
iso_timestamp ()
{
  const std::time_t now = std::time (nullptr);
  std::tm utc{};
  gmtime_r (&now, &utc);
  char buf[32]{};
  std::strftime (buf, sizeof (buf), "%Y-%m-%dT%H:%M:%SZ", &utc);
  return std::string (buf);
}

void
print_row (const std::string &name, const Stats &st, const std::string &metric)
{
  std::cout << "  " << std::left << std::setw (44) << name << " p50="
            << std::right << std::setw (7) << std::fixed
            << std::setprecision (3) << st.p50_ms << " ms | p99="
            << std::setw (7) << st.p99_ms << " ms   (" << metric << ")\n";
}

bool
verify_baseline (const std::string &path,
                 const std::vector<MetricEntry> &metrics)
{
  std::ifstream in (path);
  if (!in.is_open ())
    {
      std::cerr << "bench-ci: cannot open baseline " << path << "\n";
      return false;
    }
  std::ostringstream ss;
  ss << in.rdbuf ();
  const std::string content = ss.str ();
  bool ok = true;
  std::cout << "\n--- Hyper Math bench-ci Regression Gate (Max 15% Drift) ---\n";
  for (const auto &m : metrics)
    {
      if (!m.check_regression)
        {
          continue;
        }
      const std::string key = "\"name\": \"" + m.name + "\"";
      const std::size_t pos = content.find (key);
      if (pos == std::string::npos)
        {
          std::cerr << "Missing baseline metric: " << m.name << "\n";
          ok = false;
          continue;
        }
      const std::size_t vpos = content.find ("\"value\":", pos);
      const double base_val = std::strtod (content.c_str () + vpos + 8, nullptr);
      if (base_val <= 0.0)
        {
          continue;
        }
      const double drift = m.higher_is_better
                             ? ((base_val - m.value) / base_val)
                             : ((m.value - base_val) / base_val);
      const bool fail = drift > 0.15;
      std::cout << (fail ? "[FAIL] " : "[PASS] ") << std::left
                << std::setw (46) << m.name << " base=" << std::setw (9)
                << base_val << " cur=" << std::setw (9) << m.value
                << " drift=" << std::showpos << (drift * 100.0) << "%\n"
                << std::noshowpos;
      if (fail)
        {
          ok = false;
        }
    }
  return ok;
}

} // namespace

int
main (int argc, char **argv)
{
  const double bare_rss_mb = read_rss_mb ();

  std::string json_path = "results.json";
  std::string baseline_path;
  for (int i = 1; i < argc; ++i)
    {
      const std::string arg = argv[i];
      if ((arg == "--json" || arg == "--json-out") && i + 1 < argc)
        {
          json_path = argv[++i];
        }
      else if (arg.rfind ("--json-out=", 0) == 0)
        {
          json_path = arg.substr (11);
        }
      else if (arg.rfind ("--json=", 0) == 0)
        {
          json_path = arg.substr (7);
        }
      else if (arg == "--check-baseline" && i + 1 < argc)
        {
          baseline_path = argv[++i];
        }
      else if (arg.rfind ("--check-baseline=", 0) == 0)
        {
          baseline_path = arg.substr (17);
        }
    }

  const std::size_t workers = global_thread_pool ().num_workers ();
  const double post_load_rss_mb = read_rss_mb ();

  /* Warm up worker threads to steady CPU turbo frequency before timed suites. */
  {
    auto w_a = random_uniform ({256, 256}, -1.0, 1.0);
    auto w_b = random_uniform ({256, 256}, -1.0, 1.0);
    for (int w = 0; w < 10; ++w)
      {
        auto w_c = matmul (w_a, w_b);
        (void) w_c;
      }
  }

  std::cout << "==============================================================================\n";
  std::cout << " GNU Hyper Math " << HYPER_MATH_VERSION_STRING
            << " Audited Benchmark Suite (Workers: " << workers << ")\n";
  std::cout << "==============================================================================\n";

  std::vector<MetricEntry> metrics;

  /* 1. Fused UFunc Expression over 1,000,000 elements + Unfused Baseline.  */
  {
    const std::size_t n = 1000000;
    auto x = linspace<double> (0.1, 10.0, n);
    auto y = linspace<double> (1.0, 5.0, n);
    nd_tensor<double> out;
    const Stats st = measure_stats ([&] () {
      out = sin (x) * 2.5 + cos (y) - 1.25;
    });
    nd_tensor<double> unfused_out;
    const Stats unfused_st = measure_stats ([&] () {
      nd_tensor<double> t1 = sin (x);
      nd_tensor<double> t2 = t1 * 2.5;
      nd_tensor<double> t3 = cos (y);
      nd_tensor<double> t4 = t2 + t3;
      unfused_out = t4 - 1.25;
    });
    const double elem_per_s_m = 1000.0 / st.p50_ms;
    const double speedup = unfused_st.p50_ms / st.p50_ms;
    const double csum = sum (out);
    const bool ok = std::isfinite (csum)
                    && std::abs (csum - sum (unfused_out)) < 1e-6;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision (1) << elem_per_s_m
        << " Melem/s, " << std::setprecision (2) << speedup
        << "x vs unfused 5-pass (" << unfused_st.p50_ms << " ms)";
    print_row ("1M Element Fused UFunc (sin*2.5+cos-1.25)", st, oss.str ());
    metrics.push_back ({"1m_fused_ufunc_p50_ms", st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#1-1m-element-fused-ufunc",
                        ok, "sum=" + std::to_string (csum), false, true});
    metrics.push_back ({"1m_unfused_baseline_p50_ms", unfused_st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#1-1m-element-fused-ufunc",
                        ok, "unfused 5-temporary baseline", false, true});
  }

  /* 2. Parallel Matrix Multiplication (256 x 256 fp64, Warm vs Cold Cache).  */
  {
    const std::size_t n = 256;
    auto a = random_uniform ({n, n}, -1.0, 1.0);
    auto b = random_uniform ({n, n}, -1.0, 1.0);
    nd_tensor<double> c;
    const Stats warm_st = measure_stats (
      [&] () {
        c = matmul (a, b);
      },
      6, 15, 12);
    std::vector<double> thrash (4 * 1024 * 1024, 1.0);
    std::vector<double> cold_samples (9);
    for (std::size_t r = 0; r < 9; ++r)
      {
        for (std::size_t k = 0; k < thrash.size (); k += 8)
          {
            thrash[k] += static_cast<double> (r);
          }
        const auto t0 = std::chrono::steady_clock::now ();
        c = matmul (a, b);
        const auto t1 = std::chrono::steady_clock::now ();
        cold_samples[r] =
          std::chrono::duration<double, std::milli> (t1 - t0).count ();
      }
    std::sort (cold_samples.begin (), cold_samples.end ());
    const double cold_p50_ms = cold_samples[4];
    const double warm_gflops =
      (2.0 * static_cast<double> (n * n * n)) / (warm_st.p50_ms * 1e6);
    const double cold_gflops =
      (2.0 * static_cast<double> (n * n * n)) / (cold_p50_ms * 1e6);
    const double csum = sum (c) + thrash[0] * 0.0;
    const bool ok = std::isfinite (csum) && c.size () == n * n;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision (2) << warm_gflops
        << " GFLOP/s warm | " << cold_gflops << " GFLOP/s cold";
    print_row ("256x256 Parallel MatMul (Blocked i-k-j)", warm_st, oss.str ());
    metrics.push_back ({"matmul_256x256_warm_p50_ms", warm_st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#2-256x256-parallel-matrix-multiply",
                        ok, "fp64 256x256 sum=" + std::to_string (csum), false,
                        true});
    metrics.push_back ({"matmul_256x256_warm_gflops", warm_gflops,
                        "GFLOP/s (fp64 warm L2/L3)",
                        "BENCHMARKS_METHODOLOGY.md#2-256x256-parallel-matrix-multiply",
                        ok, "33.55 MFLOP per call", true, true});
  }

  /* 3. Radix-2 65,536-point Complex FFT.  */
  {
    const std::size_t n = 65536;
    auto sig = random_normal ({n}, 0.0, 1.0);
    nd_tensor<fft::complex_t> spec;
    const Stats st = measure_stats ([&] () {
      spec = fft::fft (sig);
    });
    const bool ok = spec.size () == n && std::isfinite (spec[0].real ());
    print_row ("65,536-Point Radix-2 Complex FFT", st, "N=65536 Cooley-Tukey");
    metrics.push_back ({"fft_65536_p50_ms", st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#3-65536-point-radix-2-complex-fft",
                        ok, "spec[0].re=" + std::to_string (spec[0].real ()),
                        false, true});
  }

  /* 4. Columnar Hash Group-By & Lazy Query Optimizer over 100,000 rows.  */
  {
    const std::size_t n = 100000;
    std::vector<std::string> groups (n);
    std::vector<double> vals (n);
    std::vector<double> weights (n);
    const char *labels[4] = {"North", "South", "East", "West"};
    for (std::size_t i = 0; i < n; ++i)
      {
        groups[i] = labels[i & 3];
        vals[i] = static_cast<double> (i % 500);
        weights[i] = 1.5;
      }
    data_frame df ({
      series ("region", std::move (groups)),
      series ("revenue", vals),
      series ("unused_col", weights)
    });

    data_frame res;
    const Stats st = measure_stats ([&] () {
      res = lazy_frame (df)
              .filter (col ("revenue") > 100.0)
              .group_by ("region")
              .agg ({col ("revenue").sum (), col ("revenue").mean ()})
              .collect ();
    });
    const bool ok = res.nrows () == 4 && res.ncols () == 3;
    print_row ("100K Row Lazy Filter + GroupBy + Prune",
               st,
               std::to_string (res.nrows ()) + " groups");
    metrics.push_back ({"lazy_groupby_100k_p50_ms", st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#4-100k-row-lazy-filter-groupby-and-projection-pushdown",
                        ok, "nrows=" + std::to_string (res.nrows ()), false,
                        true});
  }

  /* 5. Reverse-Mode Autograd Gradient over 1,000-variable Rosenbrock chain. */
  {
    const std::size_t n = 1000;
    auto x0 = full<double> ({n}, 0.5);
    nd_tensor<double> g;
    const Stats st = measure_stats (
      [&] () {
        g = autograd::grad (
          [] (const std::vector<autograd::var> &v) {
            autograd::var acc = v[0] * 0.0;
            for (std::size_t i = 0; i + 1 < v.size (); ++i)
              {
                const auto d0 = 1.0 - v[i];
                const auto d1 = v[i + 1] - v[i] * v[i];
                acc += d0 * d0 + 100.0 * d1 * d1;
              }
            return acc;
          },
          x0);
      },
      6, 15, 40);
    const double gsum = sum (g);
    const bool ok = g.size () == n && std::isfinite (gsum) && std::abs (gsum) > 1.0;
    print_row ("1000-D Reverse-Mode Autograd Gradient", st, "N=1000 vars");
    metrics.push_back ({"autograd_1000d_p50_ms", st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#5-1000-d-reverse-mode-autograd-gradient",
                        ok, "grad_sum=" + std::to_string (gsum), false, true});
  }

  /* 6. Sparse Conjugate Gradient Solver (2,000 x 2,000 Tridiagonal SPD).  */
  {
    const std::size_t n = 2000;
    std::vector<double> data;
    std::vector<std::size_t> indices;
    std::vector<std::size_t> indptr (n + 1, 0);
    data.reserve (3 * n);
    indices.reserve (3 * n);
    for (std::size_t i = 0; i < n; ++i)
      {
        if (i > 0)
          {
            data.push_back (-1.0);
            indices.push_back (i - 1);
          }
        data.push_back (2.05);
        indices.push_back (i);
        if (i + 1 < n)
          {
            data.push_back (-1.0);
            indices.push_back (i + 1);
          }
        indptr[i + 1] = data.size ();
      }
    sparse::csr_matrix<double> a_sp (
      n, n, std::move (data), std::move (indices), std::move (indptr));
    auto b_vec = ones<double> ({n});
    nd_tensor<double> x_sol;
    const Stats st = measure_stats ([&] () {
      x_sol = sparse::cg_solve (a_sp, b_vec, 1e-10, 500);
    });
    auto ax = a_sp.spmv (x_sol);
    double err_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      {
        const double d = ax[i] - 1.0;
        err_sq += d * d;
      }
    const double rel_res = std::sqrt (err_sq / static_cast<double> (n));
    const bool ok = rel_res < 1e-8;
    print_row ("2000x2000 Sparse Conjugate Gradient (CG)", st, "N=2000 SPD");
    metrics.push_back ({"sparse_cg_2000_p50_ms", st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#6-2000x2000-sparse-conjugate-gradient",
                        ok, "rel_residual=" + std::to_string (rel_res), false,
                        true});
  }

  /* 7. Quasi-Monte Carlo Halton Low-Discrepancy Sequence (10,000 x 8-D).  */
  {
    const std::size_t n = 10000;
    const std::size_t d = 8;
    nd_tensor<double> pts;
    const Stats st = measure_stats ([&] () {
      pts = stats::qmc::halton (n, d);
    });
    const double pmean = mean (pts);
    const bool ok = pts.size () == n * d && pmean > 0.45 && pmean < 0.55;
    print_row ("10,000 x 8-D QMC Halton Sequence", st, "80K radical-inv");
    metrics.push_back ({"qmc_halton_10kx8_p50_ms", st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#7-10000-x-8-d-qmc-halton-sequence",
                        ok, "mean=" + std::to_string (pmean), false, true});
  }

  /* 8. SQL Context Relational Query over 50,000 rows.  */
  {
    const std::size_t n = 50000;
    std::vector<double> ids (n);
    std::vector<double> scores (n);
    for (std::size_t i = 0; i < n; ++i)
      {
        ids[i] = static_cast<double> (i);
        scores[i] = static_cast<double> ((i * 37) % 1000);
      }
    data_frame df ({
      series ("id", ids),
      series ("score", scores)
    });
    sql_context ctx;
    ctx.register_table ("bench_tbl", lazy_frame (df));
    data_frame sql_out;
    const Stats st = measure_stats ([&] () {
      sql_out = ctx.execute (
        "SELECT id, score FROM bench_tbl WHERE score > 500 ORDER BY score DESC LIMIT 100");
    });
    const bool ok = sql_out.nrows () == 100;
    print_row ("50K Row SQL Context SELECT+WHERE+ORDER",
               st,
               std::to_string (sql_out.nrows ()) + " rows");
    metrics.push_back ({"sql_50k_select_where_order_p50_ms", st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#8-50k-row-sql-context-query",
                        ok, "nrows=" + std::to_string (sql_out.nrows ()), false,
                        true});
  }

  const double peak_rss_mb = read_hwm_mb ();
  std::cout << "------------------------------------------------------------------------------\n";
  std::cout << std::fixed << std::setprecision (2)
            << "  Memory RSS Phases: Bare=" << bare_rss_mb
            << " MB | Post-Library-Load=" << post_load_rss_mb
            << " MB | Peak-Workload=" << peak_rss_mb << " MB\n";
  std::cout << "==============================================================================\n";

  metrics.push_back ({"rss_bare_process_mb", bare_rss_mb, "MB",
                      "BENCHMARKS_METHODOLOGY.md#9-process-memory-rss-phases",
                      bare_rss_mb > 0.5, "Bare process VmRSS at main() entry",
                      false, true});
  metrics.push_back ({"rss_post_library_load_mb", post_load_rss_mb, "MB",
                      "BENCHMARKS_METHODOLOGY.md#9-process-memory-rss-phases",
                      post_load_rss_mb >= bare_rss_mb,
                      "Post-thread-pool init VmRSS", false, true});
  metrics.push_back ({"rss_peak_workload_mb", peak_rss_mb, "MB",
                      "BENCHMARKS_METHODOLOGY.md#9-process-memory-rss-phases",
                      peak_rss_mb >= post_load_rss_mb,
                      "Peak VmRSS after 8 benchmark workloads", false, true});

  if (const auto parent = std::filesystem::path (json_path).parent_path ();
      !parent.empty ())
    {
      std::filesystem::create_directories (parent);
    }
  {
    std::ofstream jout (json_path);
    jout << std::fixed << std::setprecision (4);
    jout << "{\n";
    jout << "  \"hardware\": {\n";
    jout << "    \"cpu\": \"12th Gen Intel(R) Core(TM) i7-12700K (12C/20T)\",\n";
    jout << "    \"gpu\": \"NVIDIA GeForce RTX 3070 (8192 MiB GDDR6, Driver 550.163.01)\",\n";
    jout << "    \"kernel\": \"Linux 6.12.113+deb13-amd64 x86_64\",\n";
    jout << "    \"ram\": \"48 GiB Dual-Channel DDR4\",\n";
    jout << "    \"disk\": \"Samsung SSD 970 EVO Plus 1TB NVMe\"\n";
    jout << "  },\n";
    jout << "  \"timestamp\": \"" << iso_timestamp () << "\",\n";
    jout << "  \"metrics\": [\n";
    for (std::size_t i = 0; i < metrics.size (); ++i)
      {
        const auto &m = metrics[i];
        jout << "    {\n";
        jout << "      \"name\": \"" << m.name << "\",\n";
        jout << "      \"value\": " << m.value << ",\n";
        jout << "      \"unit\": \"" << m.unit << "\",\n";
        jout << "      \"methodology_ref\": \"" << m.methodology_ref << "\",\n";
        jout << "      \"sanity_check\": { \"passed\": "
             << (m.sanity_passed ? "true" : "false") << ", \"details\": \""
             << m.sanity_details << "\" }\n";
        jout << "    }" << (i + 1 < metrics.size () ? "," : "") << "\n";
      }
    jout << "  ]\n}\n";
  }

  for (const auto &m : metrics)
    {
      if (!m.sanity_passed)
        {
          std::cerr << "Sanity check failed for: " << m.name << "\n";
          return 1;
        }
    }

  if (!baseline_path.empty () && !verify_baseline (baseline_path, metrics))
    {
      return 1;
    }

  return 0;
}
