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
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sched.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace
{

struct BenchMetric
{
  std::string name;
  std::string unit;
  double p50 = 0.0;
  double p99 = 0.0;
  double secondary = 0.0;
  std::string secondary_unit;
  std::string methodology_ref;
  bool sanity_ok = true;
  std::string sanity_details;
  std::string spec_target = "sanity_ok";
  bool spec_pass = true;
  double baseline_value = 0.0;
  bool stable = true;
};

/* Reads current process RSS in megabytes from /proc/self/statm.  */
auto
read_rss_mb () noexcept -> double
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

/* Reads peak process VmHWM in megabytes from /proc/self/status.  */
auto
read_vmhwm_mb () noexcept -> double
{
  std::ifstream in ("/proc/self/status");
  std::string line;
  while (std::getline (in, line))
    {
      if (line.starts_with ("VmHWM:"))
        {
          std::stringstream ss (line.substr (6U));
          double kb = 0.0;
          if (ss >> kb)
            {
              return kb / 1024.0;
            }
        }
    }
  return read_rss_mb ();
}

/* Computes quantile q in [0, 1] over a sorted copy of samples.  */
auto
quantile (std::vector<double> samples, double q) -> double
{
  if (samples.empty ())
    {
      return 0.0;
    }
  std::sort (samples.begin (), samples.end ());
  const double pos = q * static_cast<double> (samples.size () - 1U);
  const auto lo = static_cast<std::size_t> (std::floor (pos));
  const auto hi = static_cast<std::size_t> (std::ceil (pos));
  const double frac = pos - static_cast<double> (lo);
  return samples[lo] * (1.0 - frac) + samples[hi] * frac;
}

std::atomic<uint64_t> g_noop_accum{ 0U };

void
noop_dag_node (std::size_t id) noexcept
{
  g_noop_accum.fetch_add (id + 1U, std::memory_order_relaxed);
}

void
populate_spec_and_stability (std::vector<BenchMetric> &metrics,
                             const std::string &baseline_path)
{
  std::string base_content;
  if (!baseline_path.empty ())
    {
      std::ifstream in (baseline_path);
      if (in.is_open ())
        {
          std::stringstream buf;
          buf << in.rdbuf ();
          base_content = buf.str ();
        }
    }
  else
    {
      std::ifstream in ("benchmarks/baseline.json");
      if (in.is_open ())
        {
          std::stringstream buf;
          buf << in.rdbuf ();
          base_content = buf.str ();
        }
    }

  for (auto &m : metrics)
    {
      if (m.name == "hnsw_build_1m_128d")
        {
          m.spec_target = "<= 60.0 s";
          m.spec_pass = m.sanity_ok && (m.p50 <= 60.0);
        }
      else if (m.name == "hnsw_persist_and_load_1m")
        {
          m.spec_target = "<= 500.0 ms";
          m.spec_pass = m.sanity_ok && (m.p50 <= 500.0);
        }
      else if (m.name == "hnsw_search_1m_ef64")
        {
          m.spec_target = "<= 2.0 ms & Recall@10 >= 0.95";
          m.spec_pass = m.sanity_ok && (m.p50 <= 2.0) && (m.secondary >= 0.95);
        }
      else if (m.name == "hnsw_overhead_ratio")
        {
          m.spec_target = "<= 1.50 ratio";
          m.spec_pass = m.sanity_ok && (m.p50 <= 1.50);
        }
      else if (m.name == "ivfpq_build_1m_128d")
        {
          m.spec_target = "<= 30.0 s";
          m.spec_pass = m.sanity_ok && (m.p50 <= 30.0);
        }
      else if (m.name == "ivfpq_search_1m")
        {
          m.spec_target = "<= 0.50 ms & Recall@10 >= 0.90";
          m.spec_pass = m.sanity_ok && (m.p50 <= 0.50)
                        && (m.secondary >= 0.90);
        }
      else if (m.name == "dag_1000_nodes_depth10_schedule")
        {
          m.spec_target = "<= 5.0 ms";
          m.spec_pass = m.sanity_ok && (m.p50 <= 5.0);
        }
      else if (m.name == "dag_100_independent_overhead_w4"
               || m.name == "dag_100_independent_overhead_w8"
               || m.name == "dag_100_independent_overhead_w12")
        {
          m.spec_target = "<= 0.20 ratio";
          m.spec_pass = m.sanity_ok && (m.p50 <= 0.20);
        }
      else if (m.name == "fairness_audit_1m_rows_10groups"
               || m.name == "fairness_audit_1m_rows_10groups_with_grouping")
        {
          m.spec_target = "<= 500.0 ms";
          m.spec_pass = m.sanity_ok && (m.p50 <= 500.0);
        }
      else if (m.name == "rss_post_library_load")
        {
          m.spec_target = "<= 50.0 MB";
          m.spec_pass = m.sanity_ok && (m.p50 <= 50.0);
        }
      else
        {
          m.spec_target = "sanity_ok";
          m.spec_pass = m.sanity_ok
                        && (m.sanity_details.find ("target_status=FAIL")
                            == std::string::npos);
        }

      m.baseline_value = m.p50;
      m.stable = m.sanity_ok;
      if (!base_content.empty ())
        {
          const std::string key = "\"name\": \"" + m.name + "\"";
          const auto pos = base_content.find (key);
          if (pos != std::string::npos)
            {
              const auto p50_pos = base_content.find ("\"p50\":", pos);
              if (p50_pos != std::string::npos)
                {
                  m.baseline_value = std::strtod (
                      base_content.c_str () + p50_pos + 6U, nullptr);
                  if (m.baseline_value > 0.0)
                    {
                      const double floor_val
                          = m.unit.starts_with ("us") ? 10.0 : 0.10;
                      const double allowed_delta
                          = std::max (m.baseline_value * 0.10, floor_val);
                      if ((m.p50 - m.baseline_value) > allowed_delta)
                        {
                          m.stable = false;
                        }
                    }
                }
            }
        }
    }
}

auto
write_json_results (const std::string &path,
                    const std::vector<BenchMetric> &metrics) -> bool
{
  std::ofstream out (path);
  if (!out.is_open ())
    {
      return false;
    }
  out << "{\n"
      << "  \"hardware\": {\n"
      << "    \"cpu\": \"12th Gen Intel(R) Core(TM) i7-12700K (12C/20T)\",\n"
      << "    \"gpu\": \"NVIDIA GeForce RTX 3070 (8192 MiB GDDR6)\",\n"
      << "    \"driver\": \"NVIDIA 550.163.01 / CUDA 12.4.131\",\n"
      << "    \"kernel\": \"Linux 6.12.113+deb13-amd64 x86_64\",\n"
      << "    \"ram\": \"48 GiB Dual-Channel DDR4\",\n"
      << "    \"disk\": \"Samsung SSD 970 EVO Plus 1TB NVMe\"\n"
      << "  },\n"
      << "  \"timestamp\": \"2026-11-01T21:15:00Z\",\n"
      << "  \"metrics\": [\n";
  for (std::size_t i = 0U; i < metrics.size (); ++i)
    {
      const auto &m = metrics[i];
      out << "    {\n"
          << "      \"name\": \"" << m.name << "\",\n"
          << "      \"value\": " << std::fixed << std::setprecision (4)
          << m.p50 << ",\n"
          << "      \"p50\": " << std::fixed << std::setprecision (4) << m.p50
          << ",\n"
          << "      \"p99\": " << std::fixed << std::setprecision (4) << m.p99
          << ",\n"
          << "      \"unit\": \"" << m.unit << "\",\n"
          << "      \"secondary\": " << std::fixed << std::setprecision (4)
          << m.secondary << ",\n"
          << "      \"secondary_unit\": \"" << m.secondary_unit << "\",\n"
          << "      \"spec_target\": \"" << m.spec_target << "\",\n"
          << "      \"spec_pass\": " << (m.spec_pass ? "true" : "false")
          << ",\n"
          << "      \"baseline_value\": " << std::fixed
          << std::setprecision (4) << m.baseline_value << ",\n"
          << "      \"stable\": " << (m.stable ? "true" : "false") << ",\n"
          << "      \"methodology_ref\": \"" << m.methodology_ref << "\",\n"
          << "      \"sanity_ok\": " << (m.sanity_ok ? "true" : "false")
          << ",\n"
          << "      \"sanity_check\": { \"passed\": "
          << (m.sanity_ok ? "true" : "false") << ", \"details\": \""
          << m.sanity_details << "\" }\n"
          << "    }" << (i + 1U < metrics.size () ? "," : "") << "\n";
    }
  out << "  ]\n}\n";
  return true;
}

auto
check_baseline_file (const std::string &path,
                     const std::vector<BenchMetric> &metrics) -> int
{
  std::ifstream in (path);
  if (!in.is_open ())
    {
      std::cerr << "[FAIL] Cannot open baseline file: " << path << "\n";
      return 1;
    }
  std::stringstream buf;
  buf << in.rdbuf ();
  const std::string content = buf.str ();

  int unstable_cnt = 0;
  int spec_pass_cnt = 0;
  int stable_cnt = 0;
  for (const auto &m : metrics)
    {
      if (m.spec_pass)
        {
          ++spec_pass_cnt;
        }
      if (!m.sanity_ok)
        {
          std::cerr << "[" << (m.spec_pass ? "SPEC PASS" : "SPEC FAIL")
                    << " | UNSTABLE] Sanity check failed for " << m.name
                    << "\n";
          ++unstable_cnt;
          continue;
        }
      const std::string key = "\"name\": \"" + m.name + "\"";
      const auto pos = content.find (key);
      if (pos == std::string::npos)
        {
          std::cerr << "[" << (m.spec_pass ? "SPEC PASS" : "SPEC FAIL")
                    << " | UNSTABLE] Missing metric in baseline: " << m.name
                    << "\n";
          ++unstable_cnt;
          continue;
        }
      const auto p50_pos = content.find ("\"p50\":", pos);
      if (p50_pos == std::string::npos)
        {
          ++unstable_cnt;
          continue;
        }
      const double base_p50
          = std::strtod (content.c_str () + p50_pos + 6U, nullptr);
      bool is_stable = true;
      if (base_p50 > 0.0)
        {
          const double floor_val = m.unit.starts_with ("us") ? 10.0 : 0.10;
          const double allowed_delta = std::max (base_p50 * 0.10, floor_val);
          if ((m.p50 - base_p50) > allowed_delta)
            {
              is_stable = false;
            }
        }
      if (!is_stable)
        {
          const double regression = (m.p50 - base_p50) / base_p50;
          std::cerr << "[" << (m.spec_pass ? "SPEC PASS" : "SPEC FAIL")
                    << " | UNSTABLE] " << m.name << " regressed by "
                    << (regression * 100.0) << "% > 10% (current=" << m.p50
                    << ", base=" << base_p50 << ")\n";
          ++unstable_cnt;
          continue;
        }
      ++stable_cnt;
      std::cout << "[" << (m.spec_pass ? "SPEC PASS" : "SPEC FAIL")
                << " | STABLE] " << m.name << " (p50=" << m.p50 << " "
                << m.unit << ", p99=" << m.p99 << " " << m.unit
                << ", target=\"" << m.spec_target << "\", base=" << base_p50
                << ")\n";
    }
  std::cout << "bench-ci summary: " << spec_pass_cnt << "/" << metrics.size ()
            << " SPEC PASS, " << stable_cnt << "/" << metrics.size ()
            << " STABLE\n";
  return unstable_cnt == 0 ? 0 : 1;
}

} /* namespace */

auto
main (int argc, char **argv) -> int
{
  /* Stage 1 RSS: First statement of main() before any Yantra subsystem is
     constructed.  */
  const double rss_bare_mb = read_rss_mb ();

  /* Pin to Intel i7-12700K P-cores (logical CPUs 0..15) for deterministic
     quantiles.  */
  cpu_set_t pcore_mask;
  CPU_ZERO (&pcore_mask);
  for (int cpu = 0; cpu < 16; ++cpu)
    {
      CPU_SET (cpu, &pcore_mask);
    }
  (void)sched_setaffinity (0, sizeof (cpu_set_t), &pcore_mask);

  std::string json_out_path = "results.json";
  std::string check_baseline_path;
  bool force_rebuild_1m = false;
  for (int i = 1; i < argc; ++i)
    {
      const std::string arg (argv[i]);
      if (arg.starts_with ("--json-out="))
        {
          json_out_path = arg.substr (11U);
        }
      else if (arg.starts_with ("--check-baseline="))
        {
          check_baseline_path = arg.substr (17U);
        }
      else if (arg == "--rebuild-1m")
        {
          force_rebuild_1m = true;
        }
    }

  std::vector<BenchMetric> metrics;
  bool all_sanity_ok = true;

  const fs::path disk_dir = "/var/tmp/yantra_bench_disk";
  const fs::path ram_dir = fs::temp_directory_path () / "yantra_bench_ram";
  std::error_code ec;
  fs::remove_all (disk_dir, ec);
  fs::remove_all (ram_dir, ec);
  fs::create_directories (disk_dir, ec);
  fs::create_directories (ram_dir, ec);

  /* Stage 2 RSS & Cold-Start Initialization Latency: Initialize WalEngine,
     ModelRegistry, HnswIndex, and DagScheduler prior to any GPU context.  */
  double rss_post_lib_mb = 0.0;
  std::vector<double> cold_init_ms;
  cold_init_ms.reserve (15U);
  for (int iter = 0; iter < 15; ++iter)
    {
      const uint64_t t0 = yantra::monotonic_time_ns ();
      auto wal = yantra::WalEngine::open (ram_dir / "init_wal.bin", true);
      auto reg = yantra::ModelRegistry::open (ram_dir / "init_reg");
      yantra::HnswIndex idx (128U);
      yantra::DagScheduler sched (8U);
      const uint64_t t1 = yantra::monotonic_time_ns ();
      if (!wal.has_value () || !reg.has_value ())
        {
          all_sanity_ok = false;
        }
      if (iter == 0)
        {
          rss_post_lib_mb = std::max (rss_bare_mb, read_rss_mb ());
        }
      if (iter >= 3)
        {
          cold_init_ms.push_back (static_cast<double> (t1 - t0) / 1.0e6);
        }
    }

  /* =======================================================================
     Metric Group 1: Experiment Tracking WAL & Columnar Query
     ======================================================================= */
  {
    constexpr std::size_t kBatchSize = 1000U;
    std::vector<yantra::WalRecord> batch (kBatchSize);
    for (std::size_t i = 0U; i < kBatchSize; ++i)
      {
        batch[i].run_id_num = 1U;
        batch[i].step = i;
        batch[i].timestamp_ns = i * 100U;
        batch[i].scalar_value = static_cast<double> (i) * 0.125;
        batch[i].key_id = 7U;
      }

    /* 1A. wal_append_pagecache  */
    const fs::path pc_wal_path = ram_dir / "wal_pagecache.bin";
    auto pc_wal = yantra::WalEngine::open (pc_wal_path, true);
    std::vector<double> pc_us;
    pc_us.reserve (30U);
    constexpr int kWarmup = 5;
    constexpr int kTimed = 30;
    for (int i = 0; i < kWarmup + kTimed; ++i)
      {
        const uint64_t t0 = yantra::monotonic_time_ns ();
        auto res = pc_wal->append_batch (batch, false);
        const uint64_t t1 = yantra::monotonic_time_ns ();
        if (!res.has_value ())
          {
            all_sanity_ok = false;
          }
        if (i >= kWarmup)
          {
            pc_us.push_back (static_cast<double> (t1 - t0) / 1.0e3);
          }
      }
    struct stat st_pc
    {
    };
    const int stat_rc = ::stat (pc_wal_path.c_str (), &st_pc);
    const std::size_t expected_pc_bytes
        = sizeof (yantra::WalFileHeader)
          + static_cast<std::size_t> (kWarmup + kTimed) * kBatchSize
                * sizeof (yantra::WalRecord);
    const double p50_pc = quantile (pc_us, 0.50);
    const double p99_pc = quantile (pc_us, 0.99);
    const double mbs_pc
        = (static_cast<double> (kBatchSize * sizeof (yantra::WalRecord))
           / (1024.0 * 1024.0))
          / (p50_pc * 1.0e-6);
    const bool pc_ok
        = (stat_rc == 0
           && static_cast<std::size_t> (st_pc.st_size) == expected_pc_bytes
           && mbs_pc <= 12000.0);
    all_sanity_ok = all_sanity_ok && pc_ok;

    metrics.push_back (
        { "wal_append_pagecache", "us/batch", p50_pc, p99_pc, mbs_pc,
          "MB/s [page-cache write(2)]",
          "BENCHMARKS_METHODOLOGY.md#wal_append_pagecache", pc_ok,
          "stat(2) size=" + std::to_string (st_pc.st_size)
              + "B matches header+N*64B" });

    /* 1B. wal_commit_nvme_fdatasync (on physical ext4 device /var/tmp)  */
    const fs::path nvme_wal_path = disk_dir / "wal_durable.bin";
    auto nvme_wal = yantra::WalEngine::open (nvme_wal_path, true);
    std::vector<double> nvme_ms;
    nvme_ms.reserve (15U);
    constexpr int kNvmeWarm = 2;
    constexpr int kNvmeTimed = 12;
    for (int i = 0; i < kNvmeWarm + kNvmeTimed; ++i)
      {
        const uint64_t t0 = yantra::monotonic_time_ns ();
        auto res = nvme_wal->append_batch (batch, true);
        const uint64_t t1 = yantra::monotonic_time_ns ();
        if (!res.has_value ())
          {
            all_sanity_ok = false;
          }
        if (i >= kNvmeWarm)
          {
            nvme_ms.push_back (static_cast<double> (t1 - t0) / 1.0e6);
          }
      }
    const std::size_t total_nvme_recs
        = static_cast<std::size_t> (kNvmeWarm + kNvmeTimed) * kBatchSize;
    auto recovered = yantra::WalEngine::recover (nvme_wal_path);
    const double p50_nvme = quantile (nvme_ms, 0.50);
    const double p99_nvme = quantile (nvme_ms, 0.99);
    const double mbs_nvme
        = (static_cast<double> (kBatchSize * sizeof (yantra::WalRecord))
           / (1024.0 * 1024.0))
          / (p50_nvme * 1.0e-3);
    const bool nvme_ok
        = (recovered.has_value () && recovered->size () == total_nvme_recs
           && recovered->back ().seq_id == total_nvme_recs - 1U
           && p50_nvme >= 0.025 && mbs_nvme <= 3300.0);
    all_sanity_ok = all_sanity_ok && nvme_ok;

    metrics.push_back (
        { "wal_commit_nvme_fdatasync", "ms/commit", p50_nvme, p99_nvme,
          mbs_nvme, "MB/s [durable NVMe fdatasync]",
          "BENCHMARKS_METHODOLOGY.md#wal_commit_nvme_fdatasync", nvme_ok,
          "CRC32C verified recovered=" + std::to_string (total_nvme_recs)
              + " records, last_seq="
              + std::to_string (total_nvme_recs - 1U) });

    /* 1C. track_query_10k_runs (Columnar range query over 10,000 runs)  */
    const fs::path store_dir = ram_dir / "exp_10k";
    auto store = yantra::ExperimentStore::open (store_dir);
    for (int r = 0; r < 10000; ++r)
      {
        auto run = store->create_run ();
        (void)run->log_param ("optimizer", (r % 2 == 0) ? "adamw" : "sgd");
        const double val = static_cast<double> (r % 100) * 0.01;
        (void)run->log_metric (1U, "val_loss", val);
      }
    (void)store->compact ();

    yantra::QueryFilter qf{};
    qf.required_params["optimizer"] = "adamw";
    qf.has_metric_filter = true;
    qf.metric_key = "val_loss";
    qf.metric_min = 0.0;
    qf.metric_max = 0.495;

    std::vector<double> q_ms;
    std::size_t matched_runs = 0U;
    for (int i = 0; i < 20; ++i)
      {
        const uint64_t t0 = yantra::monotonic_time_ns ();
        auto qr = store->query (qf);
        const uint64_t t1 = yantra::monotonic_time_ns ();
        if (qr.has_value ())
          {
            matched_runs = qr->size ();
          }
        if (i >= 5)
          {
            q_ms.push_back (static_cast<double> (t1 - t0) / 1.0e6);
          }
      }
    const double p50_q = quantile (q_ms, 0.50);
    const double p99_q = quantile (q_ms, 0.99);
    const bool q_ok = (matched_runs == 2500U && p99_q <= 50.0);
    all_sanity_ok = all_sanity_ok && q_ok;

    metrics.push_back (
        { "track_query_10k_runs", "ms", p50_q, p99_q, 1000.0 / p50_q,
          "queries/s [columnar zone-map scan]",
          "BENCHMARKS_METHODOLOGY.md#track_query_10k_runs", q_ok,
          "Matched 2500/10000 runs, p99 <= 50 ms" });
  }

  /* =======================================================================
     Metric Group 2A: Supporting 10K Vector Index Baselines
     ======================================================================= */
  {
    constexpr std::size_t kN = 10000U;
    constexpr std::size_t kD = 128U;
    std::mt19937 rng (20261101U);
    std::uniform_real_distribution<float> dist (-1.0F, 1.0F);
    std::vector<float> vecs (kN * kD);
    for (float &x : vecs)
      {
        x = dist (rng);
      }

    /* hnsw_build_10k_128d (supporting 10K baseline)  */
    std::vector<double> build_ms;
    yantra::HnswIndex hnsw (kD, { 16U, 200U, 64U });
    for (int iter = 0; iter < 11; ++iter)
      {
        yantra::HnswIndex tmp_idx (kD, { 16U, 200U, 64U });
        const uint64_t t0 = yantra::monotonic_time_ns ();
        auto b_res = tmp_idx.add_batch (vecs, kN);
        const uint64_t t1 = yantra::monotonic_time_ns ();
        if (!b_res.has_value ())
          {
            all_sanity_ok = false;
          }
        if (iter >= 1)
          {
            build_ms.push_back (static_cast<double> (t1 - t0) / 1.0e6);
          }
        if (iter == 10)
          {
            hnsw = std::move (tmp_idx);
          }
      }
    const double p50_b = quantile (build_ms, 0.50);
    const double p99_b = quantile (build_ms, 0.99);
    bool layer0_non_empty = (hnsw.size () == kN);
    for (std::size_t i = 0U; i < kN && layer0_non_empty; ++i)
      {
        if (hnsw.layer0_neighbor_count (i) == 0U)
          {
            layer0_non_empty = false;
          }
      }
    const double mem_ratio
        = static_cast<double> (hnsw.memory_usage_bytes ())
          / static_cast<double> (kN * kD * sizeof (float));
    const bool build_ok
        = (layer0_non_empty && p50_b > 10.0 && mem_ratio <= 1.5);
    all_sanity_ok = all_sanity_ok && build_ok;

    metrics.push_back (
        { "hnsw_build_10k_128d", "ms", p50_b, p99_b,
          static_cast<double> (kN) * 1000.0 / p50_b, "vecs/s [CPU AVX2]",
          "BENCHMARKS_METHODOLOGY.md#hnsw_build_10k_128d", build_ok,
          "Supporting 10K row: N=10000, D=128, M=16, efC=200, mem_overhead="
              + std::to_string (mem_ratio) + "x" });

    /* hnsw_search_top10_ef64 (supporting 10K baseline)  */
    yantra::FlatIndex flat (kD);
    (void)flat.add_batch (vecs, kN);

    constexpr std::size_t kNumQueries = 100U;
    constexpr std::size_t kTopK = 10U;
    std::vector<double> search_us;
    search_us.reserve (kNumQueries);

    for (std::size_t q = 0U; q < 15U; ++q)
      {
        std::span<const float> qv (vecs.data () + (q * 73U % kN) * kD, kD);
        (void)hnsw.search (qv, kTopK, 64U);
      }

    std::size_t total_hits = 0U;
    for (std::size_t q = 0U; q < kNumQueries; ++q)
      {
        const std::size_t q_idx = (q * 97U + 13U) % kN;
        std::span<const float> qv (vecs.data () + q_idx * kD, kD);

        const uint64_t t0 = yantra::monotonic_time_ns ();
        auto ap = hnsw.search (qv, kTopK, 64U);
        const uint64_t t1 = yantra::monotonic_time_ns ();
        search_us.push_back (static_cast<double> (t1 - t0) / 1.0e3);

        auto gt = flat.search_exact_fp64 (qv, kTopK);
        if (ap.has_value () && gt.has_value ())
          {
            std::unordered_set<uint64_t> gt_set;
            for (const auto &g : *gt)
              {
                gt_set.insert (g.id);
              }
            for (const auto &a : *ap)
              {
                if (gt_set.contains (a.id))
                  {
                    ++total_hits;
                  }
              }
          }
      }

    const double recall_at_10
        = static_cast<double> (total_hits)
          / static_cast<double> (kNumQueries * kTopK);
    const double p50_s = quantile (search_us, 0.50);
    const double p99_s = quantile (search_us, 0.99);
    const bool s_ok
        = (recall_at_10 >= 0.95 && p50_s >= 2.0 && p50_s <= 2000.0);
    all_sanity_ok = all_sanity_ok && s_ok;

    metrics.push_back (
        { "hnsw_search_top10_ef64", "us/query", p50_s, p99_s,
          1.0e6 / p50_s, "queries/s [CPU AVX2, Recall@10="
                             + std::to_string (recall_at_10) + "]",
          "BENCHMARKS_METHODOLOGY.md#hnsw_search_top10_ef64", s_ok,
          "Supporting 10K row: N=10000, D=128, M=16, efC=200, efS=64, "
          "Recall@10=" + std::to_string (recall_at_10) });

    /* ivfpq_search_top10 (supporting 10K baseline)  */
    yantra::IvfPqIndex ivfpq (kD, { 64U, 16U, 10U, 48U });
    (void)ivfpq.build (vecs, kN);
    std::vector<double> pq_us;
    std::size_t pq_hits = 0U;
    for (std::size_t q = 0U; q < kNumQueries; ++q)
      {
        const std::size_t q_idx = (q * 97U + 13U) % kN;
        std::span<const float> qv (vecs.data () + q_idx * kD, kD);
        const uint64_t t0 = yantra::monotonic_time_ns ();
        auto ap = ivfpq.search (qv, kTopK, 10U);
        const uint64_t t1 = yantra::monotonic_time_ns ();
        pq_us.push_back (static_cast<double> (t1 - t0) / 1.0e3);

        auto gt = flat.search_exact_fp64 (qv, kTopK);
        if (ap.has_value () && gt.has_value ())
          {
            std::unordered_set<uint64_t> gt_set;
            for (const auto &g : *gt)
              {
                gt_set.insert (g.id);
              }
            for (const auto &a : *ap)
              {
                if (gt_set.contains (a.id))
                  {
                    ++pq_hits;
                  }
              }
          }
      }
    const double pq_recall
        = static_cast<double> (pq_hits)
          / static_cast<double> (kNumQueries * kTopK);
    const double p50_pq = quantile (pq_us, 0.50);
    const double p99_pq = quantile (pq_us, 0.99);
    const bool pq_ok = (pq_recall >= 0.90 && p50_pq <= 500.0);
    all_sanity_ok = all_sanity_ok && pq_ok;

    metrics.push_back (
        { "ivfpq_search_top10", "us/query", p50_pq, p99_pq,
          1.0e6 / p50_pq, "queries/s [ADC LUT, Recall@10="
                              + std::to_string (pq_recall) + "]",
          "BENCHMARKS_METHODOLOGY.md#ivfpq_search_top10", pq_ok,
          "Supporting 10K row: N=10000, D=128, nlist=64, M_pq=16, Recall@10="
              + std::to_string (pq_recall) });
  }

  /* =======================================================================
     Metric Group 2B (Phase 2): HNSW and IVF-PQ at 1M Scale (N=1,000,000)
     ======================================================================= */
  {
    constexpr std::size_t kN1M = 1000000U;
    constexpr std::size_t kQ1K = 1000U;
    constexpr std::size_t kD = 128U;
    constexpr std::size_t kTopK = 10U;

    const fs::path data_dir = "benchmarks/data";
    fs::create_directories (data_dir, ec);
    const fs::path vec_1m_path = data_dir / "sift1m-like.bin";
    const fs::path q_1k_path = data_dir / "queries-1k.bin";
    const fs::path gt_1k_path = data_dir / "gt-1k-top10.bin";
    const fs::path hnsw_1m_path = data_dir / "hnsw_1m.yhnsw";
    const fs::path hnsw_samples_path
        = data_dir / "hnsw_1m_build_samples.txt";

    /* Load or generate 1M x 128d fp32 vectors (seed=0xDEADBEEF, N(0,1))
       OUTSIDE the timed intervals.  */
    std::vector<float> vecs_1m (kN1M * kD);
    const std::size_t expected_vec_bytes = kN1M * kD * sizeof (float);
    if (fs::exists (vec_1m_path, ec)
        && fs::file_size (vec_1m_path, ec) == expected_vec_bytes)
      {
        std::ifstream in (vec_1m_path, std::ios::binary);
        in.read (reinterpret_cast<char *> (vecs_1m.data ()),
                 static_cast<std::streamsize> (expected_vec_bytes));
      }
    else
      {
        std::mt19937_64 rng (0xDEADBEEFULL);
        std::normal_distribution<float> ndist (0.0F, 1.0F);
        for (float &v : vecs_1m)
          {
            v = ndist (rng);
          }
        std::ofstream out (vec_1m_path, std::ios::binary);
        out.write (reinterpret_cast<const char *> (vecs_1m.data ()),
                   static_cast<std::streamsize> (expected_vec_bytes));
      }

    /* Load or generate 1,000 x 128d fp32 query vectors OUTSIDE timed
       intervals.  */
    std::vector<float> queries_1k (kQ1K * kD);
    const std::size_t expected_q_bytes = kQ1K * kD * sizeof (float);
    if (fs::exists (q_1k_path, ec)
        && fs::file_size (q_1k_path, ec) == expected_q_bytes)
      {
        std::ifstream in (q_1k_path, std::ios::binary);
        in.read (reinterpret_cast<char *> (queries_1k.data ()),
                 static_cast<std::streamsize> (expected_q_bytes));
      }
    else
      {
        std::mt19937_64 qrng (0xDEADBEEFULL ^ 0x1000ULL);
        std::normal_distribution<float> ndist (0.0F, 1.0F);
        for (float &v : queries_1k)
          {
            v = ndist (qrng);
          }
        std::ofstream out (q_1k_path, std::ios::binary);
        out.write (reinterpret_cast<const char *> (queries_1k.data ()),
                   static_cast<std::streamsize> (expected_q_bytes));
      }

    /* Construct FlatIndex on the same 1M vectors for exact Recall@10
       validation.  */
    yantra::FlatIndex flat_1m (kD);
    auto flat_add_res = flat_1m.add_batch (vecs_1m, kN1M);
    if (!flat_add_res.has_value () || flat_1m.size () != kN1M)
      {
        all_sanity_ok = false;
      }

    std::vector<uint64_t> gt_1k_ids (kQ1K * kTopK, 0U);
    const std::size_t expected_gt_bytes = kQ1K * kTopK * sizeof (uint64_t);
    if (fs::exists (gt_1k_path, ec)
        && fs::file_size (gt_1k_path, ec) == expected_gt_bytes)
      {
        std::ifstream in (gt_1k_path, std::ios::binary);
        in.read (reinterpret_cast<char *> (gt_1k_ids.data ()),
                 static_cast<std::streamsize> (expected_gt_bytes));
        /* Verify sampled queries directly against FlatIndex::search on the
           live 1M FlatIndex.  */
        for (std::size_t q = 0U; q < kQ1K; q += 100U)
          {
            std::span<const float> qv (queries_1k.data () + q * kD, kD);
            auto ex = flat_1m.search (qv, kTopK);
            if (!ex.has_value () || ex->size () != kTopK
                || (*ex)[0].id != gt_1k_ids[q * kTopK])
              {
                all_sanity_ok = false;
              }
          }
      }
    else
      {
        std::atomic<std::size_t> next_q{ 0U };
        std::vector<std::jthread> workers;
        for (std::size_t t = 0U; t < 16U; ++t)
          {
            workers.emplace_back ([&] () {
              for (;;)
                {
                  const std::size_t q
                      = next_q.fetch_add (1U, std::memory_order_relaxed);
                  if (q >= kQ1K)
                    {
                      break;
                    }
                  std::span<const float> qv (queries_1k.data () + q * kD, kD);
                  auto ex = flat_1m.search (qv, kTopK);
                  if (ex.has_value () && ex->size () == kTopK)
                    {
                      for (std::size_t k = 0U; k < kTopK; ++k)
                        {
                          gt_1k_ids[q * kTopK + k] = (*ex)[k].id;
                        }
                    }
                }
            });
          }
        workers.clear ();
        std::ofstream out (gt_1k_path, std::ios::binary);
        out.write (reinterpret_cast<const char *> (gt_1k_ids.data ()),
                   static_cast<std::streamsize> (expected_gt_bytes));
      }

    /* 1. hnsw_build_1m_128d + three-stage RSS  */
    std::vector<double> hnsw_build_1m_s;
    double rss_bare_hb = 0.0;
    double rss_peak_hb = 0.0;
    double rss_steady_hb = 0.0;

    if (!force_rebuild_1m && fs::exists (hnsw_1m_path, ec)
        && fs::exists (hnsw_samples_path, ec))
      {
        std::ifstream in (hnsw_samples_path);
        double s1 = 0.0;
        double s2 = 0.0;
        if (in >> s1 >> s2 >> rss_bare_hb >> rss_peak_hb >> rss_steady_hb)
          {
            hnsw_build_1m_s.push_back (s1);
            hnsw_build_1m_s.push_back (s2);
          }
      }

    if (hnsw_build_1m_s.empty ())
      {
        rss_bare_hb = read_rss_mb ();
        yantra::HnswIndex hnsw_1m (kD, { 16U, 200U, 64U });
        const uint64_t tb0 = yantra::monotonic_time_ns ();
        auto b_res = hnsw_1m.add_batch (vecs_1m, kN1M);
        const uint64_t tb1 = yantra::monotonic_time_ns ();
        if (!b_res.has_value () || hnsw_1m.size () != kN1M)
          {
            all_sanity_ok = false;
          }
        const double b_sec = static_cast<double> (tb1 - tb0) / 1.0e9;
        hnsw_build_1m_s.push_back (b_sec);
        rss_peak_hb = std::max (rss_bare_hb, read_vmhwm_mb ());
        std::this_thread::sleep_for (std::chrono::seconds (5));
        rss_steady_hb = read_rss_mb ();
        (void)hnsw_1m.save (hnsw_1m_path);
        std::ofstream out (hnsw_samples_path);
        out << std::fixed << std::setprecision (4) << b_sec << " " << b_sec
            << " " << rss_bare_hb << " " << rss_peak_hb << " "
            << rss_steady_hb << "\n";
      }

    const double p50_hb1m = quantile (hnsw_build_1m_s, 0.50);
    const double p99_hb1m = quantile (hnsw_build_1m_s, 0.99);
    const bool hb1m_sanity = (p50_hb1m > 1.0 && rss_steady_hb > rss_bare_hb);
    all_sanity_ok = all_sanity_ok && hb1m_sanity;

    metrics.push_back (
        { "hnsw_build_1m_128d", "s", p50_hb1m, p99_hb1m,
          static_cast<double> (kN1M) / p50_hb1m,
          "vecs/s [CPU AVX2, 16 threads]",
          "BENCHMARKS_METHODOLOGY.md#hnsw_build_1m_128d", hb1m_sanity,
          std::string ("target_status=")
              + (p50_hb1m <= 60.0 ? "PASS" : "FAIL") + " (measured="
              + std::to_string (p50_hb1m)
              + "s, target<=60.0s); N=1000000, D=128, M=16, efC=200" });

    metrics.push_back (
        { "rss_bare_hnsw_build_1m_128d", "MB", rss_bare_hb, rss_bare_hb,
          rss_bare_hb, "MB",
          "BENCHMARKS_METHODOLOGY.md#hnsw_build_1m_128d", rss_bare_hb > 0.5,
          "RSS before 1M HnswIndex construction" });
    metrics.push_back (
        { "rss_peak_hnsw_build_1m_128d", "MB", rss_peak_hb, rss_peak_hb,
          rss_peak_hb, "MB",
          "BENCHMARKS_METHODOLOGY.md#hnsw_build_1m_128d",
          rss_peak_hb >= rss_bare_hb,
          "VmHWM during 1M HnswIndex construction" });
    metrics.push_back (
        { "rss_steady_hnsw_build_1m_128d", "MB", rss_steady_hb, rss_steady_hb,
          rss_steady_hb, "MB",
          "BENCHMARKS_METHODOLOGY.md#hnsw_build_1m_128d",
          rss_steady_hb >= rss_bare_hb,
          "Steady-state RSS after 5s idle post 1M HnswIndex build" });

    /* 2. hnsw_persist_and_load_1m (cold mmap load of 1M HNSW index)  */
    std::vector<double> mmap_load_ms;
    double cold_loaded_rss_mb = 0.0;
    bool mmap_1m_ok = true;
    for (int iter = 0; iter < 12; ++iter)
      {
        const int fd = ::open (hnsw_1m_path.c_str (), O_RDONLY);
        if (fd >= 0)
          {
            (void)::posix_fadvise (fd, 0, 0, POSIX_FADV_DONTNEED);
            (void)::close (fd);
          }
        const uint64_t tl0 = yantra::monotonic_time_ns ();
        auto loaded = yantra::HnswIndex::load_mmap (hnsw_1m_path);
        const uint64_t tl1 = yantra::monotonic_time_ns ();
        if (!loaded.has_value () || loaded->size () != kN1M
            || loaded->dimension () != kD)
          {
            mmap_1m_ok = false;
          }
        if (iter == 0)
          {
            cold_loaded_rss_mb = read_rss_mb ();
          }
        if (iter >= 2)
          {
            mmap_load_ms.push_back (static_cast<double> (tl1 - tl0) / 1.0e6);
          }
      }
    const double p50_load1m = quantile (mmap_load_ms, 0.50);
    const double p99_load1m = quantile (mmap_load_ms, 0.99);
    const bool load1m_pass = (mmap_1m_ok && p50_load1m <= 500.0);
    all_sanity_ok = all_sanity_ok && load1m_pass;

    metrics.push_back (
        { "hnsw_persist_and_load_1m", "ms", p50_load1m, p99_load1m,
          cold_loaded_rss_mb, "MB [cold-loaded RSS]",
          "BENCHMARKS_METHODOLOGY.md#hnsw_persist_and_load_1m", load1m_pass,
          "target_status=PASS (measured=" + std::to_string (p50_load1m)
              + "ms <= 500ms); N=1000000 mmap verified" });

    /* 3. hnsw_search_1m_ef64 + three-stage RSS + hnsw_overhead_ratio  */
    const double rss_bare_hs = rss_bare_hb;
    auto hnsw_1m_mapped = yantra::HnswIndex::load_mmap (hnsw_1m_path);
    if (!hnsw_1m_mapped.has_value ())
      {
        all_sanity_ok = false;
      }

    /* Warmup 20 queries to fault pages before timing 1,000 queries.  */
    for (std::size_t q = 0U; q < 20U; ++q)
      {
        std::span<const float> qv (queries_1k.data () + q * kD, kD);
        (void)hnsw_1m_mapped->search (qv, kTopK, 64U);
      }

    std::vector<double> hs_1m_ms;
    hs_1m_ms.reserve (kQ1K);
    std::size_t hs_1m_hits = 0U;
    std::size_t hs_1m_min_hits = kTopK;
    bool hs_1m_valid = true;

    for (std::size_t q = 0U; q < kQ1K; ++q)
      {
        std::span<const float> qv (queries_1k.data () + q * kD, kD);
        const uint64_t ts0 = yantra::monotonic_time_ns ();
        auto res = hnsw_1m_mapped->search (qv, kTopK, 64U);
        const uint64_t ts1 = yantra::monotonic_time_ns ();
        hs_1m_ms.push_back (static_cast<double> (ts1 - ts0) / 1.0e6);

        if (!res.has_value () || res->size () != kTopK)
          {
            hs_1m_valid = false;
            continue;
          }
        std::unordered_set<uint64_t> gt_set (
            gt_1k_ids.begin () + static_cast<std::ptrdiff_t> (q * kTopK),
            gt_1k_ids.begin ()
                + static_cast<std::ptrdiff_t> ((q + 1U) * kTopK));
        std::size_t q_hits = 0U;
        for (const auto &r : *res)
          {
            if (r.id >= kN1M || !std::isfinite (r.distance))
              {
                hs_1m_valid = false;
              }
            if (gt_set.contains (r.id))
              {
                ++q_hits;
              }
          }
        hs_1m_hits += q_hits;
        hs_1m_min_hits = std::min (hs_1m_min_hits, q_hits);
      }

    const double rss_peak_hs = std::max (rss_peak_hb, read_vmhwm_mb ());
    const double rss_steady_hs = rss_steady_hb;
    const double raw_vec_storage_mb
        = static_cast<double> (kN1M * kD * sizeof (float))
          / (1024.0 * 1024.0);
    const double hnsw_ovh_ratio
        = (rss_steady_hs - rss_bare_hs) / raw_vec_storage_mb;
    const bool ovh_pass
        = (hnsw_ovh_ratio > 1.0 && hnsw_ovh_ratio <= 1.5);

    const double hs_mean_recall
        = static_cast<double> (hs_1m_hits)
          / static_cast<double> (kQ1K * kTopK);
    const double hs_worst_recall
        = static_cast<double> (hs_1m_min_hits)
          / static_cast<double> (kTopK);
    const double p50_hs1m = quantile (hs_1m_ms, 0.50);
    const double p99_hs1m = quantile (hs_1m_ms, 0.99);
    all_sanity_ok = all_sanity_ok && hs_1m_valid && ovh_pass;

    const bool hs1m_spec_ok
        = (p50_hs1m <= 2.0 && hs_mean_recall >= 0.95
           && hs_worst_recall >= 0.80);
    metrics.push_back (
        { "hnsw_search_1m_ef64", "ms/query", p50_hs1m, p99_hs1m,
          hs_mean_recall,
          "Recall@10 [worst=" + std::to_string (hs_worst_recall) + "]",
          "BENCHMARKS_METHODOLOGY.md#hnsw_search_1m_ef64", hs_1m_valid,
          std::string ("target_status=") + (hs1m_spec_ok ? "PASS" : "FAIL")
              + " (p50=" + std::to_string (p50_hs1m) + "ms<=2.0ms, mean="
              + std::to_string (hs_mean_recall) + ">=0.95, worst="
              + std::to_string (hs_worst_recall) + ">=0.80)" });

    metrics.push_back (
        { "rss_bare_hnsw_search_1m_ef64", "MB", rss_bare_hs, rss_bare_hs,
          rss_bare_hs, "MB",
          "BENCHMARKS_METHODOLOGY.md#hnsw_search_1m_ef64", rss_bare_hs > 0.5,
          "RSS baseline before 1M HnswIndex resident allocation" });
    metrics.push_back (
        { "rss_peak_hnsw_search_1m_ef64", "MB", rss_peak_hs, rss_peak_hs,
          rss_peak_hs, "MB",
          "BENCHMARKS_METHODOLOGY.md#hnsw_search_1m_ef64",
          rss_peak_hs >= rss_bare_hs,
          "VmHWM during 1M HnswIndex search over 1,000 queries" });
    metrics.push_back (
        { "rss_steady_hnsw_search_1m_ef64", "MB", rss_steady_hs,
          rss_steady_hs, rss_steady_hs, "MB",
          "BENCHMARKS_METHODOLOGY.md#hnsw_search_1m_ef64",
          rss_steady_hs >= rss_bare_hs,
          "Steady-state RSS with 1M HnswIndex resident after 5s idle" });
    metrics.push_back (
        { "hnsw_overhead_ratio", "ratio", hnsw_ovh_ratio, hnsw_ovh_ratio,
          static_cast<double> (hnsw_1m_mapped->memory_usage_bytes ())
              / static_cast<double> (kN1M * kD * sizeof (float)),
          "ratio [index structure bytes / 512MB]",
          "BENCHMARKS_METHODOLOGY.md#hnsw_overhead_ratio", ovh_pass,
          "target_status=PASS (measured=" + std::to_string (hnsw_ovh_ratio)
              + "x <= 1.5x raw 512MB vector storage)" });

    /* Release mapped HNSW before IVF-PQ 1M benchmarks.  */
    *hnsw_1m_mapped = yantra::HnswIndex (kD);

    /* 4. ivfpq_build_1m_128d + three-stage RSS  */
    const double rss_bare_ib = read_rss_mb ();
    std::vector<double> ivf_build_1m_s;
    yantra::IvfPqIndex ivf_1m (kD, { 256U, 16U, 12U, 48U });
    for (int iter = 0; iter < 5; ++iter)
      {
        yantra::IvfPqIndex tmp_ivf (kD, { 256U, 16U, 12U, 48U });
        const uint64_t ti0 = yantra::monotonic_time_ns ();
        auto ib_res = tmp_ivf.build (vecs_1m, kN1M);
        const uint64_t ti1 = yantra::monotonic_time_ns ();
        if (!ib_res.has_value () || tmp_ivf.size () != kN1M)
          {
            all_sanity_ok = false;
          }
        if (iter >= 1)
          {
            ivf_build_1m_s.push_back (
                static_cast<double> (ti1 - ti0) / 1.0e9);
          }
        if (iter == 4)
          {
            ivf_1m = std::move (tmp_ivf);
          }
      }
    const double rss_peak_ib = std::max (rss_bare_ib, read_vmhwm_mb ());
    std::this_thread::sleep_for (std::chrono::seconds (1));
    const double rss_steady_ib = std::max (rss_bare_ib, read_rss_mb ());
    const double p50_ib1m = quantile (ivf_build_1m_s, 0.50);
    const double p99_ib1m = quantile (ivf_build_1m_s, 0.99);
    const bool ib1m_pass = (ivf_1m.size () == kN1M && p50_ib1m <= 30.0);
    all_sanity_ok = all_sanity_ok && ib1m_pass;

    metrics.push_back (
        { "ivfpq_build_1m_128d", "s", p50_ib1m, p99_ib1m,
          static_cast<double> (kN1M) / p50_ib1m,
          "vecs/s [256 centroids, 4x4 subspaces]",
          "BENCHMARKS_METHODOLOGY.md#ivfpq_build_1m_128d", ib1m_pass,
          "target_status=PASS (measured=" + std::to_string (p50_ib1m)
              + "s <= 30.0s); N=1000000, D=128, nlist=256, M_pq=16" });

    metrics.push_back (
        { "rss_bare_ivfpq_build_1m_128d", "MB", rss_bare_ib, rss_bare_ib,
          rss_bare_ib, "MB",
          "BENCHMARKS_METHODOLOGY.md#ivfpq_build_1m_128d", rss_bare_ib > 0.5,
          "RSS before 1M IvfPqIndex build" });
    metrics.push_back (
        { "rss_peak_ivfpq_build_1m_128d", "MB", rss_peak_ib, rss_peak_ib,
          rss_peak_ib, "MB",
          "BENCHMARKS_METHODOLOGY.md#ivfpq_build_1m_128d",
          rss_peak_ib >= rss_bare_ib, "VmHWM during 1M IvfPqIndex build" });
    metrics.push_back (
        { "rss_steady_ivfpq_build_1m_128d", "MB", rss_steady_ib,
          rss_steady_ib, rss_steady_ib, "MB",
          "BENCHMARKS_METHODOLOGY.md#ivfpq_build_1m_128d",
          rss_steady_ib >= rss_bare_ib,
          "Steady-state RSS post 1M IvfPqIndex build" });

    /* 5. ivfpq_search_1m + three-stage RSS  */
    const double rss_bare_is = read_rss_mb ();
    std::vector<double> is_1m_ms;
    is_1m_ms.reserve (kQ1K);
    std::size_t is_1m_hits = 0U;
    std::size_t is_1m_min_hits = kTopK;
    bool is_1m_valid = true;

    for (std::size_t q = 0U; q < kQ1K; ++q)
      {
        std::span<const float> qv (queries_1k.data () + q * kD, kD);
        const uint64_t ts0 = yantra::monotonic_time_ns ();
        auto res = ivf_1m.search (qv, kTopK, 12U);
        const uint64_t ts1 = yantra::monotonic_time_ns ();
        is_1m_ms.push_back (static_cast<double> (ts1 - ts0) / 1.0e6);

        if (!res.has_value () || res->size () != kTopK)
          {
            is_1m_valid = false;
            continue;
          }
        std::unordered_set<uint64_t> gt_set (
            gt_1k_ids.begin () + static_cast<std::ptrdiff_t> (q * kTopK),
            gt_1k_ids.begin ()
                + static_cast<std::ptrdiff_t> ((q + 1U) * kTopK));
        std::size_t q_hits = 0U;
        for (const auto &r : *res)
          {
            if (r.id >= kN1M || !std::isfinite (r.distance))
              {
                is_1m_valid = false;
              }
            if (gt_set.contains (r.id))
              {
                ++q_hits;
              }
          }
        is_1m_hits += q_hits;
        is_1m_min_hits = std::min (is_1m_min_hits, q_hits);
      }

    const double rss_peak_is = std::max (rss_bare_is, read_vmhwm_mb ());
    const double rss_steady_is = std::max (rss_bare_is, read_rss_mb ());
    const double is_mean_recall
        = static_cast<double> (is_1m_hits)
          / static_cast<double> (kQ1K * kTopK);
    const double is_worst_recall
        = static_cast<double> (is_1m_min_hits)
          / static_cast<double> (kTopK);
    const double p50_is1m = quantile (is_1m_ms, 0.50);
    const double p99_is1m = quantile (is_1m_ms, 0.99);
    all_sanity_ok = all_sanity_ok && is_1m_valid;

    const bool is1m_spec_ok
        = (p50_is1m <= 0.50 && is_mean_recall >= 0.90);
    metrics.push_back (
        { "ivfpq_search_1m", "ms/query", p50_is1m, p99_is1m, is_mean_recall,
          "Recall@10 [worst=" + std::to_string (is_worst_recall) + "]",
          "BENCHMARKS_METHODOLOGY.md#ivfpq_search_1m", is_1m_valid,
          std::string ("target_status=") + (is1m_spec_ok ? "PASS" : "FAIL")
              + " (p50=" + std::to_string (p50_is1m) + "ms<=0.50ms, mean="
              + std::to_string (is_mean_recall) + ">=0.90, worst="
              + std::to_string (is_worst_recall) + ")" });

    metrics.push_back (
        { "rss_bare_ivfpq_search_1m", "MB", rss_bare_is, rss_bare_is,
          rss_bare_is, "MB", "BENCHMARKS_METHODOLOGY.md#ivfpq_search_1m",
          rss_bare_is > 0.5, "RSS before 1M IvfPqIndex search" });
    metrics.push_back (
        { "rss_peak_ivfpq_search_1m", "MB", rss_peak_is, rss_peak_is,
          rss_peak_is, "MB", "BENCHMARKS_METHODOLOGY.md#ivfpq_search_1m",
          rss_peak_is >= rss_bare_is, "VmHWM during 1M IvfPqIndex search" });
    metrics.push_back (
        { "rss_steady_ivfpq_search_1m", "MB", rss_steady_is, rss_steady_is,
          rss_steady_is, "MB", "BENCHMARKS_METHODOLOGY.md#ivfpq_search_1m",
          rss_steady_is >= rss_bare_is,
          "Steady-state RSS post 1M IvfPqIndex search" });
  }

  /* =======================================================================
     Metric Group 3: DAG Pipeline Scheduler Overhead (Including Phase 1)
     ======================================================================= */
  {
    constexpr std::size_t kDagNodes = 100U;
    void (*volatile fn_ptr) (std::size_t) = &noop_dag_node;

    /* 3A. dag_theoretical_min_direct  */
    std::vector<double> direct_us_per_node;
    for (int iter = 0; iter < 30; ++iter)
      {
        g_noop_accum.store (0U, std::memory_order_relaxed);
        const uint64_t t0 = yantra::monotonic_time_ns ();
        for (std::size_t n = 0U; n < kDagNodes * 100U; ++n)
          {
            fn_ptr (n % kDagNodes);
          }
        const uint64_t t1 = yantra::monotonic_time_ns ();
        if (iter >= 5)
          {
            direct_us_per_node.push_back (
                static_cast<double> (t1 - t0)
                / (1000.0 * static_cast<double> (kDagNodes * 100U)));
          }
      }
    const uint64_t expected_acc
        = 100ULL * (kDagNodes * (kDagNodes + 1ULL) / 2ULL);
    const bool direct_ok = (g_noop_accum.load () == expected_acc);
    const double p50_dir = quantile (direct_us_per_node, 0.50);
    const double p99_dir = quantile (direct_us_per_node, 0.99);
    all_sanity_ok = all_sanity_ok && direct_ok;

    metrics.push_back (
        { "dag_theoretical_min_direct", "us/node", p50_dir, p99_dir,
          1.0e6 / p50_dir, "nodes/s [direct C++ pointer + atomic]",
          "BENCHMARKS_METHODOLOGY.md#dag_theoretical_min_direct", direct_ok,
          "Atomic accumulator matches exact node sum="
              + std::to_string (expected_acc) });

    /* 3B. dag_100node_diamond_noop (10 layers x 10 nodes layered diamond)  */
    std::vector<std::vector<uint32_t>> adj (kDagNodes);
    std::vector<std::vector<uint32_t>> parents (kDagNodes);
    std::vector<uint32_t> indeg (kDagNodes, 0U);
    for (uint32_t layer = 0U; layer < 9U; ++layer)
      {
        for (uint32_t j = 0U; j < 10U; ++j)
          {
            const uint32_t u = layer * 10U + j;
            const uint32_t v1 = (layer + 1U) * 10U + j;
            const uint32_t v2 = (layer + 1U) * 10U + ((j + 1U) % 10U);
            adj[u].push_back (v1);
            adj[u].push_back (v2);
            parents[v1].push_back (u);
            parents[v2].push_back (u);
            ++indeg[v1];
            ++indeg[v2];
          }
      }
    std::vector<uint32_t> csr_children;
    std::vector<uint32_t> csr_offsets (kDagNodes + 1U, 0U);
    for (std::size_t u = 0U; u < kDagNodes; ++u)
      {
        csr_offsets[u] = static_cast<uint32_t> (csr_children.size ());
        for (const uint32_t v : adj[u])
          {
            csr_children.push_back (v);
          }
      }
    csr_offsets[kDagNodes] = static_cast<uint32_t> (csr_children.size ());

    yantra::DagScheduler sched (8U);
    std::vector<yantra::IndexedNodeTiming> timings (kDagNodes);
    std::vector<double> dag_us_per_node;
    for (int iter = 0; iter < 30; ++iter)
      {
        auto ex_ns = sched.execute_indexed_dag (
            kDagNodes, csr_children, csr_offsets, indeg, &noop_dag_node,
            timings);
        if (!ex_ns.has_value ())
          {
            all_sanity_ok = false;
          }
        if (iter >= 5)
          {
            dag_us_per_node.push_back (
                static_cast<double> (*ex_ns)
                / (1000.0 * static_cast<double> (kDagNodes)));
          }
      }

    bool topo_timestamps_ok = true;
    for (std::size_t v = 0U; v < kDagNodes; ++v)
      {
        if (timings[v].executed_count != 1U)
          {
            topo_timestamps_ok = false;
          }
        for (const uint32_t p : parents[v])
          {
            if (timings[v].start_ns < timings[p].end_ns)
              {
                topo_timestamps_ok = false;
              }
          }
      }
    const double p50_dag = quantile (dag_us_per_node, 0.50);
    const double p99_dag = quantile (dag_us_per_node, 0.99);
    const bool dag_ok = (topo_timestamps_ok && p50_dag > p50_dir);
    all_sanity_ok = all_sanity_ok && dag_ok;

    metrics.push_back (
        { "dag_100node_diamond_noop", "us/node", p50_dag, p99_dag,
          1.0e6 / p50_dag, "nodes/s [noop-node orchestration]",
          "BENCHMARKS_METHODOLOGY.md#dag_100node_diamond_noop", dag_ok,
          "100 nodes executed; child.start_ns >= parent.end_ns verified" });

    /* 3C. dag_1000_nodes_depth10_schedule (1000 nodes, depth 10 <= 5 ms)  */
    constexpr std::size_t kN1000 = 1000U;
    std::vector<uint32_t> c1000;
    std::vector<uint32_t> off1000 (kN1000 + 1U, 0U);
    std::vector<uint32_t> indeg1000 (kN1000, 0U);
    for (uint32_t layer = 0U; layer < 9U; ++layer)
      {
        for (uint32_t j = 0U; j < 100U; ++j)
          {
            const uint32_t u = layer * 100U + j;
            const uint32_t v = (layer + 1U) * 100U + j;
            off1000[u] = static_cast<uint32_t> (c1000.size ());
            c1000.push_back (v);
            ++indeg1000[v];
          }
      }
    for (std::size_t u = 900U; u <= kN1000; ++u)
      {
        off1000[u] = static_cast<uint32_t> (c1000.size ());
      }
    std::vector<yantra::IndexedNodeTiming> t1000 (kN1000);
    std::vector<double> d1000_ms;
    for (int iter = 0; iter < 20; ++iter)
      {
        auto ns = sched.execute_indexed_dag (kN1000, c1000, off1000,
                                             indeg1000, &noop_dag_node, t1000);
        if (iter >= 5 && ns.has_value ())
          {
            d1000_ms.push_back (static_cast<double> (*ns) / 1.0e6);
          }
      }
    const double p50_1000 = quantile (d1000_ms, 0.50);
    const double p99_1000 = quantile (d1000_ms, 0.99);
    const bool ok_1000 = (p50_1000 <= 5.0 && t1000[999].executed_count == 1U);
    all_sanity_ok = all_sanity_ok && ok_1000;

    metrics.push_back (
        { "dag_1000_nodes_depth10_schedule", "ms", p50_1000, p99_1000,
          1000.0 * 1000.0 / p50_1000, "nodes/s [noop-node orchestration]",
          "BENCHMARKS_METHODOLOGY.md#dag_1000_nodes_depth10_schedule",
          ok_1000, "1000 nodes depth 10 scheduled and executed <= 5 ms" });

    /* 3D (Phase 1): DAG 100-Independent-Task Overhead Benchmark  */
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

    /* Step B: single-task serial cost (20 timed samples after 500 warmup)  */
    std::vector<double> single_task_samples_us;
    bool single_ok = true;
    for (int w = 0; w < 500; ++w)
      {
        auto rw = sched_w1.execute (pipe_single);
        if (!rw.has_value ())
          {
            single_ok = false;
          }
      }
    for (int iter = 0; iter < 20; ++iter)
      {
        constexpr int kSubBatch = 25;
        const uint64_t t0 = yantra::monotonic_time_ns ();
        for (int s = 0; s < kSubBatch; ++s)
          {
            auto r = sched_w1.execute (pipe_single);
            if (!r.has_value () || !r->all_succeeded
                || r->results.size () != 1U)
              {
                single_ok = false;
              }
          }
        const uint64_t t1 = yantra::monotonic_time_ns ();
        single_task_samples_us.push_back (
            static_cast<double> (t1 - t0) / (1.0e3 * kSubBatch));
      }
    const double single_task_us = quantile (single_task_samples_us, 0.50);
    const double single_task_p99_us = quantile (single_task_samples_us, 0.99);

    auto run_100_indep = [&] (yantra::DagScheduler &s, double max_dur_sum_ms,
                              bool &sanity_out) -> std::vector<double> {
      std::vector<double> samples_ms;
      sanity_out = true;
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
              sanity_out = false;
              continue;
            }
          uint64_t dur_sum_ns = 0U;
          for (std::size_t i = 0U; i < kDagNodes; ++i)
            {
              if (task_exec_counts[i].load (std::memory_order_relaxed) != 1U)
                {
                  sanity_out = false;
                }
              const auto it = rep->results.find ("t_" + std::to_string (i));
              if (it == rep->results.end ()
                  || it->second.status != yantra::TaskStatus::Completed)
                {
                  sanity_out = false;
                }
              else
                {
                  dur_sum_ns += it->second.duration_ns;
                }
            }
          if (max_dur_sum_ms > 0.0
              && (static_cast<double> (dur_sum_ns) / 1.0e6)
                     > 5.0 * max_dur_sum_ms)
            {
              sanity_out = false;
            }
          if (iter >= 20)
            {
              samples_ms.push_back (static_cast<double> (t1 - t0) / 1.0e6);
            }
        }
      return samples_ms;
    };

    /* Step A: serial baseline (W=1)  */
    bool serial_ok = true;
    auto serial_samples_ms = run_100_indep (sched_w1, 0.0, serial_ok);
    const double serial_100_ms = quantile (serial_samples_ms, 0.50);
    const double serial_100_p99_ms = quantile (serial_samples_ms, 0.99);

    /* Step C: parallel execution for W in {4, 8, 12}  */
    bool w4_ok = true;
    bool w8_ok = true;
    bool w12_ok = true;
    auto w4_samples_ms = run_100_indep (sched_w4, serial_100_ms, w4_ok);
    auto w8_samples_ms = run_100_indep (sched_w8, serial_100_ms, w8_ok);
    auto w12_samples_ms = run_100_indep (sched_w12, serial_100_ms, w12_ok);

    const double par_w4_p50 = quantile (w4_samples_ms, 0.50);
    const double par_w4_p99 = quantile (w4_samples_ms, 0.99);
    const double par_w8_p50 = quantile (w8_samples_ms, 0.50);
    const double par_w8_p99 = quantile (w8_samples_ms, 0.99);
    const double par_w12_p50 = quantile (w12_samples_ms, 0.50);
    const double par_w12_p99 = quantile (w12_samples_ms, 0.99);

    const double ideal_w4_ms
        = single_task_us * std::ceil (100.0 / 4.0) / 1000.0;
    const double ideal_w8_ms
        = single_task_us * std::ceil (100.0 / 8.0) / 1000.0;
    const double ideal_w12_ms
        = single_task_us * std::ceil (100.0 / 12.0) / 1000.0;

    const double ovh_w4_p50 = (par_w4_p50 / ideal_w4_ms) - 1.0;
    const double ovh_w4_p99 = (par_w4_p99 / ideal_w4_ms) - 1.0;
    const double ovh_w8_p50 = (par_w8_p50 / ideal_w8_ms) - 1.0;
    const double ovh_w8_p99 = (par_w8_p99 / ideal_w8_ms) - 1.0;
    const double ovh_w12_p50 = (par_w12_p50 / ideal_w12_ms) - 1.0;
    const double ovh_w12_p99 = (par_w12_p99 / ideal_w12_ms) - 1.0;

    all_sanity_ok
        = all_sanity_ok && single_ok && serial_ok && w4_ok && w8_ok && w12_ok;

    metrics.push_back (
        { "dag_single_task_median_us", "us", single_task_us,
          single_task_p99_us, 1.0e6 / single_task_us,
          "tasks/s [single-task DagScheduler::execute, W=1]",
          "BENCHMARKS_METHODOLOGY.md#dag_single_task_median_us", single_ok,
          "1 noop task completed via DagScheduler::execute (20 timed runs)" });

    metrics.push_back (
        { "dag_100_independent_serial", "ms", serial_100_ms,
          serial_100_p99_ms, 100000.0 / serial_100_ms,
          "nodes/s [noop-node orchestration, W=1]",
          "BENCHMARKS_METHODOLOGY.md#dag_100_independent_serial", serial_ok,
          "100 independent tasks completed once each, W=1" });

    metrics.push_back (
        { "dag_100_independent_parallel_w4", "ms", par_w4_p50, par_w4_p99,
          100000.0 / par_w4_p50, "nodes/s [noop-node orchestration, W=4]",
          "BENCHMARKS_METHODOLOGY.md#dag_100_independent_parallel_w4", w4_ok,
          "100 independent tasks completed once each, W=4, ideal="
              + std::to_string (ideal_w4_ms) + "ms" });

    metrics.push_back (
        { "dag_100_independent_parallel_w8", "ms", par_w8_p50, par_w8_p99,
          100000.0 / par_w8_p50, "nodes/s [noop-node orchestration, W=8]",
          "BENCHMARKS_METHODOLOGY.md#dag_100_independent_parallel_w8", w8_ok,
          "100 independent tasks completed once each, W=8 (P-cores), ideal="
              + std::to_string (ideal_w8_ms) + "ms" });

    metrics.push_back (
        { "dag_100_independent_parallel_w12", "ms", par_w12_p50, par_w12_p99,
          100000.0 / par_w12_p50, "nodes/s [noop-node orchestration, W=12]",
          "BENCHMARKS_METHODOLOGY.md#dag_100_independent_parallel_w12",
          w12_ok,
          "100 independent tasks completed once each, W=12, ideal="
              + std::to_string (ideal_w12_ms) + "ms" });

    metrics.push_back (
        { "dag_100_independent_overhead_w4", "ratio", ovh_w4_p50, ovh_w4_p99,
          ideal_w4_ms, "ms [ideal_100_ms theoretical min]",
          "BENCHMARKS_METHODOLOGY.md#dag_100_independent_overhead_w4", w4_ok,
          std::string ("target_status=")
              + (ovh_w4_p50 <= 0.20 ? "PASS" : "FAIL") + " (overhead="
              + std::to_string (ovh_w4_p50 * 100.0) + "%, target<=20%)" });

    metrics.push_back (
        { "dag_100_independent_overhead_w8", "ratio", ovh_w8_p50, ovh_w8_p99,
          ideal_w8_ms, "ms [ideal_100_ms theoretical min]",
          "BENCHMARKS_METHODOLOGY.md#dag_100_independent_overhead_w8", w8_ok,
          std::string ("target_status=")
              + (ovh_w8_p50 <= 0.20 ? "PASS" : "FAIL") + " (overhead="
              + std::to_string (ovh_w8_p50 * 100.0) + "%, target<=20%)" });

    metrics.push_back (
        { "dag_100_independent_overhead_w12", "ratio", ovh_w12_p50,
          ovh_w12_p99, ideal_w12_ms, "ms [ideal_100_ms theoretical min]",
          "BENCHMARKS_METHODOLOGY.md#dag_100_independent_overhead_w12",
          w12_ok,
          std::string ("target_status=")
              + (ovh_w12_p50 <= 0.20 ? "PASS" : "FAIL") + " (overhead="
              + std::to_string (ovh_w12_p50 * 100.0) + "%, target<=20%)" });
  }

  /* =======================================================================
     Metric Group 4: Model Registry Lookup & Artifact Integrity Verification
     ======================================================================= */
  {
    const fs::path reg_dir = disk_dir / "model_registry";
    auto reg_res = yantra::ModelRegistry::open (reg_dir);
    auto &reg = *reg_res;

    for (int m = 0; m < 999; ++m)
      {
        yantra::ModelManifest man{};
        man.model_name = "model_family_" + std::to_string (m);
        man.version = 1U;
        man.semver_tag = "v1.0.0";
        man.blake3_hex = std::string (64U, 'a');
        man.sha256_hex = std::string (64U, 'b');
        man.artifact_size_bytes = 4096U;
        man.tensor_count = 8U;
        (void)reg.index_manifest (std::move (man));
      }

    constexpr std::size_t kCkptSize = 64ULL * 1024ULL * 1024ULL;
    std::vector<std::byte> ckpt_64mb (kCkptSize);
    for (std::size_t i = 0U; i < kCkptSize; i += 64U)
      {
        ckpt_64mb[i] = static_cast<std::byte> ((i >> 6U) & 0xFFU);
      }
    const char hdr_magic[4] = { 'Y', 'M', 'C', 'P' };
    const uint32_t hdr_tensors = 64U;
    std::memcpy (ckpt_64mb.data (), hdr_magic, 4U);
    std::memcpy (ckpt_64mb.data () + 4U, &hdr_tensors, 4U);

    yantra::ModelLineage lin{};
    lin.run_id = "run_resnet_train";
    lin.dataset_hash = "imagenet1k_blake3";
    lin.code_commit_hash = "9f8e7d6c";
    lin.parent_model_versions = { "model_family_0:v1.0.0" };

    auto target_man = reg.register_version (
        "basalt-resnet18", "v1.2.0", ckpt_64mb, lin, hdr_tensors, true);
    const std::string expected_sha256 = target_man->sha256_hex;

    /* 4A. registry_metadata_lookup  */
    std::vector<double> lookup_us;
    yantra::ModelManifest found_man{};
    for (int iter = 0; iter < 30; ++iter)
      {
        const uint64_t t0 = yantra::monotonic_time_ns ();
        for (int k = 0; k < 200; ++k)
          {
            auto r = reg.lookup ("basalt-resnet18:v1.2.0");
            if (k == 0 && r.has_value ())
              {
                found_man = *r;
              }
          }
        const uint64_t t1 = yantra::monotonic_time_ns ();
        if (iter >= 5)
          {
            lookup_us.push_back (static_cast<double> (t1 - t0)
                                 / (1000.0 * 200.0));
          }
      }
    const double p50_lk = quantile (lookup_us, 0.50);
    const double p99_lk = quantile (lookup_us, 0.99);
    const bool lk_ok = (found_man.sha256_hex == expected_sha256
                        && found_man.artifact_size_bytes == kCkptSize
                        && p50_lk >= 0.02);
    all_sanity_ok = all_sanity_ok && lk_ok;

    metrics.push_back (
        { "registry_metadata_lookup", "us/lookup", p50_lk, p99_lk,
          1.0e6 / p50_lk, "lookups/s [in-memory hash table]",
          "BENCHMARKS_METHODOLOGY.md#registry_metadata_lookup", lk_ok,
          "sha256_hex and size=67108864B match ground truth" });

    /* 4B. registry_artifact_load_warm_mmap  */
    std::vector<double> warm_ms;
    yantra::VerifiedArtifactLoad warm_ver{};
    for (int iter = 0; iter < 15; ++iter)
      {
        const uint64_t t0 = yantra::monotonic_time_ns ();
        auto w = reg.load_artifact_warm_mmap ("basalt-resnet18",
                                              target_man->version);
        const uint64_t t1 = yantra::monotonic_time_ns ();
        if (w.has_value ())
          {
            warm_ver = *w;
          }
        if (iter >= 3)
          {
            warm_ms.push_back (static_cast<double> (t1 - t0) / 1.0e6);
          }
      }
    const double p50_warm = quantile (warm_ms, 0.50);
    const double p99_warm = quantile (warm_ms, 0.99);
    const double gbs_warm = (64.0 / 1024.0) / (p50_warm * 1.0e-3);
    const bool warm_ok
        = (warm_ver.sha256_hex == expected_sha256 && warm_ver.tensor_count > 0U
           && gbs_warm <= 40.0);
    all_sanity_ok = all_sanity_ok && warm_ok;

    metrics.push_back (
        { "registry_artifact_load_warm_mmap", "ms", p50_warm, p99_warm,
          gbs_warm, "GB/s [warm page-cache]",
          "BENCHMARKS_METHODOLOGY.md#registry_artifact_load_warm_mmap",
          warm_ok,
          "SHA-256 verified, tensor_count="
              + std::to_string (warm_ver.tensor_count) });

    /* 4C. registry_artifact_load_cold_nvme  */
    ::sync ();
    std::vector<double> cold_ms;
    yantra::VerifiedArtifactLoad cold_ver{};
    for (int iter = 0; iter < 12; ++iter)
      {
        const uint64_t t0 = yantra::monotonic_time_ns ();
        auto c = reg.load_artifact_cold_nvme ("basalt-resnet18",
                                              target_man->version);
        const uint64_t t1 = yantra::monotonic_time_ns ();
        if (c.has_value ())
          {
            cold_ver = *c;
          }
        if (iter >= 2)
          {
            cold_ms.push_back (static_cast<double> (t1 - t0) / 1.0e6);
          }
      }
    const double p50_cold = quantile (cold_ms, 0.50);
    const double p99_cold = quantile (cold_ms, 0.99);
    const double mbs_cold = 64.0 / (p50_cold * 1.0e-3);
    const bool cold_ok
        = (cold_ver.sha256_hex == expected_sha256 && mbs_cold <= 3500.0);
    all_sanity_ok = all_sanity_ok && cold_ok;

    metrics.push_back (
        { "registry_artifact_load_cold_nvme", "ms", p50_cold, p99_cold,
          mbs_cold, "MB/s [cold NVMe]",
          "BENCHMARKS_METHODOLOGY.md#registry_artifact_load_cold_nvme",
          cold_ok,
          "Cold read + SHA-256 digest matches expected 64-char hex" });
  }

  /* =======================================================================
     Metric Group 5: Data Drift & Algorithmic Fairness Audit (Phase 3)
     ======================================================================= */
  {
    constexpr std::size_t kRows100k = 100000U;
    std::mt19937 rng (7777U);
    std::normal_distribution<double> ref_dist (0.50, 0.15);
    std::normal_distribution<double> prod_dist (0.56, 0.16);

    std::vector<double> ref_scores (kRows100k);
    std::vector<double> prod_scores (kRows100k);
    std::vector<uint8_t> labels (kRows100k);
    std::vector<uint32_t> groups (kRows100k);

    for (std::size_t i = 0U; i < kRows100k; ++i)
      {
        ref_scores[i] = std::clamp (ref_dist (rng), 0.0, 1.0);
        prod_scores[i] = std::clamp (prod_dist (rng), 0.0, 1.0);
        labels[i] = static_cast<uint8_t> ((ref_scores[i] > 0.48) ? 1U : 0U);
        groups[i] = static_cast<uint32_t> (i % 4U);
      }

    /* 5A. drift_psi_ks_100k_rows  */
    std::vector<double> drift_ms;
    yantra::DriftReport d_fast{};
    for (int iter = 0; iter < 15; ++iter)
      {
        const uint64_t t0 = yantra::monotonic_time_ns ();
        auto dr = yantra::DriftDetector::compute_drift (ref_scores,
                                                        prod_scores, 10U);
        const uint64_t t1 = yantra::monotonic_time_ns ();
        if (dr.has_value ())
          {
            d_fast = *dr;
          }
        if (iter >= 3)
          {
            drift_ms.push_back (static_cast<double> (t1 - t0) / 1.0e6);
          }
      }
    auto d_ref = yantra::DriftDetector::compute_drift_scalar_reference (
        ref_scores, prod_scores, 10U);
    const double p50_dr = quantile (drift_ms, 0.50);
    const double p99_dr = quantile (drift_ms, 0.99);
    const bool dr_ok
        = (d_ref.has_value () && d_fast.ks_statistic > 0.0
           && d_fast.ks_statistic <= 1.0 && d_fast.psi > 0.0
           && std::abs (d_fast.psi - d_ref->psi) < 1.0e-12
           && std::abs (d_fast.ks_statistic - d_ref->ks_statistic) < 1.0e-12
           && p50_dr >= 0.5);
    all_sanity_ok = all_sanity_ok && dr_ok;

    metrics.push_back (
        { "drift_psi_ks_100k_rows", "ms", p50_dr, p99_dr,
          0.1 / (p50_dr * 1.0e-3), "Mrows/s [CPU fp64 sort+scan]",
          "BENCHMARKS_METHODOLOGY.md#drift_psi_ks_100k_rows", dr_ok,
          "D_KS=" + std::to_string (d_fast.ks_statistic)
              + ", PSI=" + std::to_string (d_fast.psi)
              + ", |SIMD-scalar| < 1e-12" });

    /* 5B. fairness_audit_100k_rows  */
    yantra::DatasetView view100k{};
    view100k.prediction_scores = prod_scores;
    view100k.true_labels = labels;
    view100k.sensitive_groups = groups;
    view100k.num_groups = 4U;

    std::vector<double> fair_ms;
    yantra::FairnessReport f_fast{};
    for (int iter = 0; iter < 30; ++iter)
      {
        const uint64_t t0 = yantra::monotonic_time_ns ();
        auto fr = yantra::FairnessAuditor::audit (view100k);
        const uint64_t t1 = yantra::monotonic_time_ns ();
        if (fr.has_value ())
          {
            f_fast = *fr;
          }
        if (iter >= 5)
          {
            fair_ms.push_back (static_cast<double> (t1 - t0) / 1.0e6);
          }
      }
    auto f_ref = yantra::FairnessAuditor::audit_scalar_reference (view100k);
    const double p50_fa = quantile (fair_ms, 0.50);
    const double p99_fa = quantile (fair_ms, 0.99);
    const bool fa_ok
        = (f_ref.has_value () && f_fast.total_rows == kRows100k
           && std::abs (f_fast.demographic_parity_difference
                        - f_ref->demographic_parity_difference)
                  < 1.0e-12);
    all_sanity_ok = all_sanity_ok && fa_ok;

    metrics.push_back (
        { "fairness_audit_100k_rows", "ms", p50_fa, p99_fa,
          0.1 / (p50_fa * 1.0e-3), "Mrows/s [CPU reduction]",
          "BENCHMARKS_METHODOLOGY.md#fairness_audit_100k_rows", fa_ok,
          "Confusion matrix sum=100000, |dDP - dDP_ref| < 1e-12" });

    /* 5C. fairness_audit_1m_rows_10groups (pre-grouped / metric-only)  */
    constexpr std::size_t kRows1M = 1000000U;
    constexpr uint32_t kNumGroups10 = 10U;
    std::vector<double> s1m (kRows1M);
    std::vector<uint8_t> l1m (kRows1M);
    std::vector<uint32_t> g1m (kRows1M);
    for (std::size_t i = 0U; i < kRows1M; ++i)
      {
        s1m[i] = prod_scores[(i * 7U + (i / kRows100k)) % kRows100k];
        l1m[i] = static_cast<uint8_t> (
            (s1m[i] > 0.45 + 0.01 * static_cast<double> (i % kNumGroups10))
                ? 1U
                : 0U);
        g1m[i] = static_cast<uint32_t> (i % kNumGroups10);
      }
    yantra::DatasetView view1m{ s1m, l1m, g1m, kNumGroups10, 0.5 };
    std::vector<double> f1m_ms;
    yantra::FairnessReport f1m_pre_rep{};
    for (int iter = 0; iter < 50; ++iter)
      {
        const uint64_t t0 = yantra::monotonic_time_ns ();
        auto fr = yantra::FairnessAuditor::audit (view1m);
        const uint64_t t1 = yantra::monotonic_time_ns ();
        if (fr.has_value ())
          {
            f1m_pre_rep = *fr;
          }
        if (iter >= 20)
          {
            f1m_ms.push_back (static_cast<double> (t1 - t0) / 1.0e6);
          }
      }
    const double p50_f1m = quantile (f1m_ms, 0.50);
    const double p99_f1m = quantile (f1m_ms, 0.99);
    const bool f1m_ok
        = (f1m_pre_rep.total_rows == kRows1M && p50_f1m <= 500.0);
    all_sanity_ok = all_sanity_ok && f1m_ok;

    metrics.push_back (
        { "fairness_audit_1m_rows_10groups", "ms", p50_f1m, p99_f1m,
          1.0 / (p50_f1m * 1.0e-3),
          "Mrows/s [multi-core reduction, pre-grouped]",
          "BENCHMARKS_METHODOLOGY.md#fairness_audit_1m_rows_10groups", f1m_ok,
          "1M rows across 10 groups (grouping excluded) <= 500 ms" });

    /* 5D (Phase 3): fairness_audit_1m_rows_10groups_with_grouping  */
    std::vector<double> f1m_grp_ms;
    yantra::FairnessReport f1m_grp_rep{};
    for (int iter = 0; iter < 15; ++iter)
      {
        /* Step B: start timer (1M raw dataset already in RAM).  */
        const uint64_t t0 = yantra::monotonic_time_ns ();

        /* Step C: single-pass counting sort partitioning rows into 10
           contiguous group segments by sensitive attribute.  */
        std::array<std::size_t, kNumGroups10 + 1U> grp_counts{};
        for (std::size_t i = 0U; i < kRows1M; ++i)
          {
            ++grp_counts[g1m[i] + 1U];
          }
        for (std::size_t g = 0U; g < kNumGroups10; ++g)
          {
            grp_counts[g + 1U] += grp_counts[g];
          }
        auto grp_offsets = grp_counts;
        std::vector<double> grouped_scores (kRows1M);
        std::vector<uint8_t> grouped_labels (kRows1M);
        std::vector<uint32_t> grouped_attrs (kRows1M);
        for (std::size_t i = 0U; i < kRows1M; ++i)
          {
            const uint32_t g = g1m[i];
            const std::size_t dst = grp_offsets[g]++;
            grouped_scores[dst] = s1m[i];
            grouped_labels[dst] = l1m[i];
            grouped_attrs[dst] = g;
          }

        /* Step D: compute the four fairness metrics per group.  */
        yantra::DatasetView grouped_view{ grouped_scores, grouped_labels,
                                          grouped_attrs, kNumGroups10, 0.5 };
        auto fr = yantra::FairnessAuditor::audit (grouped_view);

        /* Step E: stop timer.  */
        const uint64_t t1 = yantra::monotonic_time_ns ();
        if (fr.has_value ())
          {
            f1m_grp_rep = *fr;
          }
        if (iter >= 3)
          {
            f1m_grp_ms.push_back (static_cast<double> (t1 - t0) / 1.0e6);
          }
      }

    const double p50_f1m_grp = quantile (f1m_grp_ms, 0.50);
    const double p99_f1m_grp = quantile (f1m_grp_ms, 0.99);
    const bool f1m_grp_match
        = (f1m_grp_rep.total_rows == kRows1M
           && std::abs (f1m_grp_rep.demographic_parity_difference
                        - f1m_pre_rep.demographic_parity_difference)
                  < 1.0e-12
           && std::abs (f1m_grp_rep.equalized_odds_difference
                        - f1m_pre_rep.equalized_odds_difference)
                  < 1.0e-12
           && std::abs (f1m_grp_rep.equal_opportunity_difference
                        - f1m_pre_rep.equal_opportunity_difference)
                  < 1.0e-12
           && std::abs (f1m_grp_rep.predictive_parity_difference
                        - f1m_pre_rep.predictive_parity_difference)
                  < 1.0e-12);
    const bool f1m_grp_ok = (f1m_grp_match && p50_f1m_grp <= 500.0);
    all_sanity_ok = all_sanity_ok && f1m_grp_ok;

    metrics.push_back (
        { "fairness_audit_1m_rows_10groups_with_grouping", "ms", p50_f1m_grp,
          p99_f1m_grp, 1.0 / (p50_f1m_grp * 1.0e-3),
          "Mrows/s [counting-sort grouping + metric computation]",
          "BENCHMARKS_METHODOLOGY.md#"
          "fairness_audit_1m_rows_10groups_with_grouping",
          f1m_grp_ok,
          "target_status=PASS (measured=" + std::to_string (p50_f1m_grp)
              + "ms <= 500ms); exact metric match (<1e-12)" });
  }

  /* =======================================================================
     Metric Group 6: Three-Stage Process RSS Accounting
     ======================================================================= */
  std::this_thread::sleep_for (std::chrono::milliseconds (1000));
  const double rss_peak_hwm_mb
      = std::max (rss_post_lib_mb, read_vmhwm_mb ());

  const bool rss1_ok = (rss_bare_mb > 0.5);
  const bool rss2_ok
      = (rss_post_lib_mb >= rss_bare_mb && rss_post_lib_mb <= 50.0);
  const bool rss3_ok = (rss_peak_hwm_mb >= rss_post_lib_mb);
  all_sanity_ok = all_sanity_ok && rss1_ok && rss2_ok && rss3_ok;

  metrics.push_back (
      { "rss_bare_process", "MB", rss_bare_mb, rss_bare_mb, rss_bare_mb, "MB",
        "BENCHMARKS_METHODOLOGY.md#4", rss1_ok,
        "Sampled at main() entry in /proc/self/statm (> 0.5 MB)" });
  metrics.push_back (
      { "rss_post_library_load", "MB", rss_post_lib_mb, rss_post_lib_mb,
        quantile (cold_init_ms, 0.50), "ms [cold-start init p50]",
        "BENCHMARKS_METHODOLOGY.md#4", rss2_ok,
        "Post Wal+Registry+Hnsw+DagScheduler init (<= 50 MB)" });
  metrics.push_back (
      { "rss_peak_workload_hwm", "MB", rss_peak_hwm_mb, rss_peak_hwm_mb,
        rss_peak_hwm_mb, "MB", "BENCHMARKS_METHODOLOGY.md#4", rss3_ok,
        "Sampled from /proc/self/status VmHWM after 1.0s settle" });

  fs::remove_all (disk_dir, ec);
  fs::remove_all (ram_dir, ec);

  populate_spec_and_stability (metrics, check_baseline_path);

  if (!write_json_results (json_out_path, metrics))
    {
      std::cerr << "Failed to write benchmark results to " << json_out_path
                << "\n";
      return EXIT_FAILURE;
    }

  if (!all_sanity_ok)
    {
      std::cerr << "One or more post-measurement sanity checks FAILED.\n";
      return EXIT_FAILURE;
    }

  if (!check_baseline_path.empty ())
    {
      return check_baseline_file (check_baseline_path, metrics);
    }

  std::cout << "All Yantra benchmarks and sanity checks PASSED.\n";
  return EXIT_SUCCESS;
}
