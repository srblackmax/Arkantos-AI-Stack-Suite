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

#include <bellows/bellows.hpp>

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
measure_stats (Fn &&fn, int warmup = 3, int repeats = 13, int batch_iters = 1)
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
print_row (const std::string &name, const Stats &st, const std::string &extra)
{
  std::cout << "  " << std::left << std::setw (46) << name << " p50="
            << std::right << std::setw (7) << std::fixed
            << std::setprecision (3) << st.p50_ms << " ms | p99="
            << std::setw (7) << st.p99_ms << " ms   (" << extra << ")\n";
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
  std::cout << "\n--- Bellows bench-ci Regression Gate (Max 15% Drift) ---\n";
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

  using namespace bellows;

  // Build deterministic tabular dataset: N=1,200 rows, D=8 features (7 continuous + 1 nominal).
  const std::size_t n_train = 1200;
  const std::size_t d_cols = 8;
  data_table x_raw (n_train, d_cols);
  x_raw.set_scale (d_cols - 1, feature_scale::nominal, 4);
  std::vector<double> y_reg (n_train, 0.0);
  std::vector<double> y_bin (n_train, 0.0);

  for (std::size_t i = 0; i < n_train; ++i)
    {
      double acc_signal = 0.0;
      for (std::size_t c = 0; c + 1 < d_cols; ++c)
        {
          const double v =
            std::sin (static_cast<double> (i * (c + 1)) * 0.017) * 2.0
            + static_cast<double> ((i + c * 13) % 29) * 0.1;
          x_raw.at (i, c) = (i % 31 == 0 && c == 1) ? missing_value () : v;
          acc_signal += (c % 2 == 0 ? 1.4 : -0.9) * v;
        }
      const double cat = static_cast<double> (i % 4);
      x_raw.at (i, d_cols - 1) = cat;
      const double cat_eff = (cat >= 2.0) ? 2.5 : -1.5;
      y_reg[i] = acc_signal + cat_eff;
      y_bin[i] = (y_reg[i] > 1.0) ? 1.0 : 0.0;
    }

  const double post_load_rss_mb = read_rss_mb ();

  // Warm up CPU frequency before timing.
  {
    column_mender wm (mending_strategy::median);
    fit (wm, x_raw);
    auto wx = transform (wm, x_raw);
    penalised_linear_regressor wlin (with_regularisation{0.0, 1.0e-3, 0.0});
    fit (wlin, wx, y_reg);
  }

  std::cout << "==============================================================================\n";
  std::cout << " Bellows 0.1.0 Audited Classical ML & GBDT Benchmark Suite\n";
  std::cout << "==============================================================================\n";

  std::vector<MetricEntry> metrics;
  explained_variance_ratio r2_metric;
  classification_accuracy acc_metric;

  // 1a. Preprocessing only (column_mender + moment_normaliser) on N=1,200 x 8.
  data_table x_clean (n_train, d_cols);
  {
    const Stats prep_st = measure_stats (
      [&] () {
        column_mender mender (mending_strategy::median);
        moment_normaliser scaler;
        fit (mender, x_raw);
        data_table mended = transform (mender, x_raw);
        fit (scaler, mended);
        x_clean = transform (scaler, mended);
      },
      4, 13, 6);
    const bool ok = x_clean.row_count () == n_train
                    && !is_missing (x_clean.at (0, 1));
    print_row ("1a. Feature Prep (Mender+Normaliser, 1200x8)", prep_st,
               "excluded from pure booster fit()");
    metrics.push_back ({"feature_prep_1200x8_p50_ms", prep_st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#1-stagewise-booster-fit-exact-vs-quantile-bin",
                        ok, "rows=1200, missing imputed", false, true});
  }

  // 1b. Stagewise Booster Fit: Quantile-Bin Policy (256 bins, 40 trees, depth 4).
  auto q_booster = make_stagewise_booster<squared_residual_loss> (
    with_booster<quantile_bin_policy,
                  best_gain_frontier_growth,
                  uniform_full_sample>{},
    with_stage_count{40},
    with_step_rate{0.15},
    with_tree_limits{4, 15, 4, 64},
    with_regularisation{0.01, 1.0, 0.0});
  {
    const Stats q_st = measure_stats (
      [&] () {
        fit (q_booster, x_clean, y_reg);
      },
      3, 11, 1);
    const double r2 = r2_metric (y_reg, predict (q_booster, x_clean));
    const bool ok = r2 > 0.85 && q_booster.tree_count () == 40;
    std::ostringstream oss;
    oss << "40 trees, 64 bins, R^2=" << std::fixed << std::setprecision (3) << r2;
    print_row ("1b. Booster Fit (quantile_bin_policy, 1200x8)", q_st, oss.str ());
    metrics.push_back ({"booster_fit_quantile_1200x8_p50_ms", q_st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#1-stagewise-booster-fit-exact-vs-quantile-bin",
                        ok, "R2=" + std::to_string (r2), false, true});
  }

  // 1c. Stagewise Booster Fit: Exact-Cut Policy (40 trees, depth 4).
  auto ex_booster = make_stagewise_booster<squared_residual_loss> (
    with_booster<exact_cut_policy,
                  depth_synchronous_growth,
                  uniform_full_sample>{},
    with_stage_count{40},
    with_step_rate{0.15},
    with_tree_limits{4, 16, 4, 64},
    with_regularisation{0.01, 1.0, 0.0});
  {
    const Stats ex_st = measure_stats (
      [&] () {
        fit (ex_booster, x_clean, y_reg);
      },
      2, 9, 1);
    const double r2 = r2_metric (y_reg, predict (ex_booster, x_clean));
    const bool ok = r2 > 0.85 && ex_booster.tree_count () == 40;
    std::ostringstream oss;
    oss << "40 trees, exact cuts, R^2=" << std::fixed << std::setprecision (3) << r2;
    print_row ("1c. Booster Fit (exact_cut_policy, 1200x8)", ex_st, oss.str ());
    metrics.push_back ({"booster_fit_exact_1200x8_p50_ms", ex_st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#1-stagewise-booster-fit-exact-vs-quantile-bin",
                        ok, "R2=" + std::to_string (r2), false, true});
  }

  // 2. Stagewise Booster Batch Predict Throughput (batch_size = 10,000, D=8, 40 trees).
  {
    const std::size_t n_batch = 10000;
    data_table x_batch (n_batch, d_cols);
    for (std::size_t i = 0; i < n_batch; ++i)
      {
        for (std::size_t c = 0; c < d_cols; ++c)
          {
            x_batch.at (i, c) = x_clean.at (i % n_train, c);
          }
      }
    std::vector<double> batch_preds;
    const Stats b_st = measure_stats (
      [&] () {
        batch_preds = predict (q_booster, x_batch);
      },
      4, 15, 6);
    const double rows_per_s =
      (static_cast<double> (n_batch) * 1000.0) / b_st.p50_ms;
    const bool ok = batch_preds.size () == n_batch
                    && std::isfinite (batch_preds[0])
                    && std::abs (batch_preds[0] - batch_preds[1]) > 1e-6;
    std::ostringstream oss;
    oss << "batch_size=10000, " << std::fixed << std::setprecision (0)
        << rows_per_s << " rows/s";
    print_row ("2. Booster Batch Predict (N=10,000, 40 trees)", b_st, oss.str ());
    metrics.push_back ({"booster_batch_predict_10k_p50_ms", b_st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#2-stagewise-booster-batch-predict-throughput",
                        ok, "batch_size=10000", false, true});
    metrics.push_back ({"booster_batch_predict_rows_per_s", rows_per_s,
                        "rows/s (batch_size=10000)",
                        "BENCHMARKS_METHODOLOGY.md#2-stagewise-booster-batch-predict-throughput",
                        ok, "N=10000 x 8 cols x 40 trees", true, true});
  }

  // 3. Single-Row Zero-Allocation Inference (infer_one, batch_size = 1).
  {
    std::vector<double> single_row (d_cols);
    for (std::size_t c = 0; c < d_cols; ++c)
      {
        single_row[c] = x_clean.at (0, c);
      }
    double acc_pred = 0.0;
    const int row_calls = 4000;
    const Stats one_st = measure_stats (
      [&] () {
        double s = 0.0;
        for (int k = 0; k < row_calls; ++k)
          {
            single_row[0] = static_cast<double> (k & 7) * 0.25;
            s += infer_one (q_booster, single_row);
          }
        acc_pred = s;
      },
      4, 15, 2);
    const double ns_per_row = (one_st.p50_ms * 1.0e6) / static_cast<double> (row_calls);
    const double p99_ns_per_row = (one_st.p99_ms * 1.0e6) / static_cast<double> (row_calls);
    const double rows_per_s = 1.0e9 / ns_per_row;
    const bool ok = std::isfinite (acc_pred) && std::abs (acc_pred) > 1.0;
    std::ostringstream oss;
    oss << "batch_size=1: p50=" << std::fixed << std::setprecision (1)
        << ns_per_row << " ns/row, p99=" << p99_ns_per_row << " ns/row ("
        << std::setprecision (0) << rows_per_s << " rows/s)";
    print_row ("3. Single-Row infer_one (4000 calls/batch)", one_st, oss.str ());
    metrics.push_back ({"infer_one_single_row_p50_ns", ns_per_row,
                        "ns/row (batch_size=1)",
                        "BENCHMARKS_METHODOLOGY.md#3-single-row-zero-allocation-inference-infer_one",
                        ok, "sum=" + std::to_string (acc_pred), false, true});
  }

  // 4. Penalised Linear Regressor & Logistic Classifier (N=1,200, D=8).
  {
    penalised_linear_regressor lin_reg (with_regularisation{0.0, 1.0e-3, 0.0});
    const Stats lin_st = measure_stats (
      [&] () {
        fit (lin_reg, x_clean, y_reg);
      },
      4, 15, 20);
    const double lin_r2 = r2_metric (y_reg, predict (lin_reg, x_clean));

    penalised_logistic_classifier log_clf (with_regularisation{0.0, 1.0e-3, 0.0});
    const Stats log_st = measure_stats (
      [&] () {
        fit (log_clf, x_clean, y_bin);
      },
      3, 13, 2);
    const double log_acc = acc_metric (y_bin, predict (log_clf, x_clean));
    const bool ok = lin_r2 > 0.70 && log_acc > 0.85;
    std::ostringstream oss1;
    oss1 << "Ridge L2 Cholesky/LU, R^2=" << std::fixed << std::setprecision (3)
         << lin_r2;
    print_row ("4a. Penalised Linear Fit (1200x8 Ridge)", lin_st, oss1.str ());
    std::ostringstream oss2;
    oss2 << "80-epoch L-BFGS/GD, Acc=" << std::fixed << std::setprecision (3)
         << log_acc;
    print_row ("4b. Penalised Logistic Fit (1200x8 Binary)", log_st, oss2.str ());
    metrics.push_back ({"penalised_linear_fit_1200x8_p50_ms", lin_st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#4-penalised-linear-and-logistic-estimators",
                        ok, "R2=" + std::to_string (lin_r2), false, true});
    metrics.push_back ({"penalised_logistic_fit_1200x8_p50_ms", log_st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#4-penalised-logistic-estimators",
                        ok, "Acc=" + std::to_string (log_acc), false, true});
  }

  // 5. Composite Pipeline (column_mender | moment_normaliser | indicator_expander | stagewise_booster).
  {
    auto pipe =
      column_mender (mending_strategy::median)
      | moment_normaliser ()
      | indicator_expander ()
      | make_stagewise_booster<squared_residual_loss> (
          with_booster<quantile_bin_policy,
                        best_gain_frontier_growth,
                        uniform_full_sample>{},
          with_stage_count{40},
          with_step_rate{0.15},
          with_tree_limits{4, 15, 4, 64},
          with_regularisation{0.01, 1.0, 0.0});
    const Stats pipe_st = measure_stats (
      [&] () {
        fit (pipe, x_raw, y_reg);
      },
      2, 11, 1);
    const double pipe_r2 = r2_metric (y_reg, predict (pipe, x_raw));
    const bool ok = pipe_r2 > 0.85;
    std::ostringstream oss;
    oss << "4-stage operator| chain, R^2=" << std::fixed
        << std::setprecision (3) << pipe_r2;
    print_row ("5. Composite Pipeline Fit (1200x8 -> 11 cols)", pipe_st,
               oss.str ());
    metrics.push_back ({"composite_pipeline_fit_1200x8_p50_ms", pipe_st.p50_ms,
                        "ms",
                        "BENCHMARKS_METHODOLOGY.md#5-composite-pipeline-end-to-end-fit",
                        ok, "R2=" + std::to_string (pipe_r2), false, true});
  }

  const double peak_rss_mb = read_hwm_mb ();
  std::cout << "------------------------------------------------------------------------------\n";
  std::cout << std::fixed << std::setprecision (2)
            << "  Memory RSS Phases: Bare=" << bare_rss_mb
            << " MB | Post-Library-Load=" << post_load_rss_mb
            << " MB | Peak-Workload=" << peak_rss_mb << " MB\n";
  std::cout << "==============================================================================\n";

  metrics.push_back ({"rss_bare_process_mb", bare_rss_mb, "MB",
                      "BENCHMARKS_METHODOLOGY.md#6-process-memory-rss-phases",
                      bare_rss_mb > 0.5, "Bare process VmRSS at main() entry",
                      false, true});
  metrics.push_back ({"rss_post_library_load_mb", post_load_rss_mb, "MB",
                      "BENCHMARKS_METHODOLOGY.md#6-process-memory-rss-phases",
                      post_load_rss_mb >= bare_rss_mb,
                      "Post-data_table init VmRSS", false, true});
  metrics.push_back ({"rss_peak_workload_mb", peak_rss_mb, "MB",
                      "BENCHMARKS_METHODOLOGY.md#6-process-memory-rss-phases",
                      peak_rss_mb >= post_load_rss_mb,
                      "Peak VmHWM across all 5 Bellows benchmark suites",
                      false, true});

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
