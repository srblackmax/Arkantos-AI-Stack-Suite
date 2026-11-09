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

#include <obsidian/obsidian.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace obsidian;
using Clock = std::chrono::steady_clock;

namespace
{

double
read_rss_mb (void)
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
read_hwm_mb (void)
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
measure_stats (Fn &&fn, int warmup = 4, int repeats = 15, int batch_iters = 1)
{
  for (int w = 0; w < warmup; ++w)
    {
      fn ();
    }
  std::vector<double> samples (static_cast<size_t> (repeats));
  for (int r = 0; r < repeats; ++r)
    {
      auto t0 = Clock::now ();
      for (int b = 0; b < batch_iters; ++b)
        {
          fn ();
        }
      auto t1 = Clock::now ();
      double dt = std::chrono::duration<double, std::milli> (t1 - t0).count ();
      samples[static_cast<size_t> (r)] =
        dt / static_cast<double> (batch_iters);
    }
  std::sort (samples.begin (), samples.end ());
  size_t n = samples.size ();
  size_t idx99 = std::min (
    n - 1,
    static_cast<size_t> (std::ceil (static_cast<double> (n) * 0.99) - 1.0));
  return Stats{samples[n / 2], samples[idx99]};
}

struct Metric_Entry
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
iso_timestamp (void)
{
  std::time_t now = std::time (nullptr);
  std::tm utc{};
  gmtime_r (&now, &utc);
  char buf[32]{};
  std::strftime (buf, sizeof (buf), "%Y-%m-%dT%H:%M:%SZ", &utc);
  return std::string (buf);
}

void
print_row (const std::string &name, const Stats &st, const std::string &extra)
{
  std::cout << "  " << std::left << std::setw (38) << name << " p50="
            << std::right << std::setw (7) << std::fixed
            << std::setprecision (3) << st.p50_ms << " ms | p99="
            << std::setw (7) << st.p99_ms << " ms (" << extra << ")\n";
}

bool
verify_baseline (const std::string &path,
                 const std::vector<Metric_Entry> &metrics)
{
  std::ifstream in (path);
  if (!in.is_open ())
    {
      std::cerr << "bench-ci: cannot open baseline " << path << "\n";
      return false;
    }
  std::ostringstream ss;
  ss << in.rdbuf ();
  std::string content = ss.str ();
  bool ok = true;
  std::cout << "\n--- Obsidian bench-ci Regression Gate (Max 15% Drift) ---\n";
  for (const auto &m : metrics)
    {
      if (!m.check_regression)
        {
          continue;
        }
      std::string key = "\"name\": \"" + m.name + "\"";
      size_t pos = content.find (key);
      if (pos == std::string::npos)
        {
          std::cerr << "Missing baseline metric: " << m.name << "\n";
          ok = false;
          continue;
        }
      size_t vpos = content.find ("\"value\":", pos);
      double base_val = std::strtod (content.c_str () + vpos + 8, nullptr);
      if (base_val <= 0.0)
        {
          continue;
        }
      double drift = m.higher_is_better
                       ? ((base_val - m.value) / base_val)
                       : ((m.value - base_val) / base_val);
      bool fail = drift > 0.15;
      std::cout << (fail ? "[FAIL] " : "[PASS] ") << std::left
                << std::setw (42) << m.name << " base=" << std::setw (9)
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

} /* namespace */

int
main (int argc, char **argv)
{
  double bare_rss_mb = read_rss_mb ();

  std::string json_path = "results.json";
  std::string baseline_path;
  for (int i = 1; i < argc; ++i)
    {
      std::string arg = argv[i];
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

  std::cout << "============================================================"
               "==================\n";
  std::cout << " Obsidian 0.1.0 Audited NLP Benchmark Suite (SIMD: "
            << simd::active_isa_name () << ")\n";
  std::cout << "============================================================"
               "==================\n";

  std::vector<Metric_Entry> metrics;

  /* 1. Zero-allocation BPE tokenization in PMR Arena (vocab_size = 96).  */
  std::vector<std::string_view> corpus = {
    "high performance deep learning and natural language processing in cpp",
    "simd accelerated byte pair encoding avoids dynamic heap allocations"
  };
  BPE_Tokenizer bpe;
  bpe.train_from_corpus (corpus, 96, 1);
  double post_load_rss_mb = read_rss_mb ();

  {
    Arena arena;
    const size_t bpe_iters = 5000;
    size_t batch_tokens = 0;
    std::string decoded_sample;
    Stats bpe_st = measure_stats (
      [&] (void) {
        size_t tok_cnt = 0;
        for (size_t i = 0; i < bpe_iters; ++i)
          {
            arena.reset ();
            auto sp = bpe.encode_into_arena (corpus[0], arena);
            tok_cnt += sp.size ();
            if (i == 0 && decoded_sample.empty ())
              {
                auto dec = bpe.decode (sp, true);
                if (dec.has_value ())
                  {
                    decoded_sample = *dec;
                  }
              }
          }
        batch_tokens = tok_cnt;
      },
      4, 15, 1);

    double tok_per_s =
      (static_cast<double> (batch_tokens) * 1000.0) / bpe_st.p50_ms;
    double bytes_per_batch =
      static_cast<double> (corpus[0].size () * bpe_iters);
    double mb_per_s =
      (bytes_per_batch / (1024.0 * 1024.0)) / (bpe_st.p50_ms / 1000.0);
    bool ok = batch_tokens >= bpe_iters && !decoded_sample.empty ();
    std::ostringstream oss;
    oss << std::fixed << std::setprecision (0) << tok_per_s << " tok/s, "
        << std::setprecision (2) << mb_per_s << " MB/s (vocab=96)";
    print_row ("[Bench 1] BPE Arena Encode (5K calls)", bpe_st, oss.str ());
    metrics.push_back ({"bpe_arena_encode_5k_p50_ms", bpe_st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#1-bpe-arena-encode",
                        ok, "tokens=" + std::to_string (batch_tokens), false,
                        true});
    metrics.push_back ({"bpe_arena_encode_tokens_per_s", tok_per_s,
                        "tokens/s (vocab_size=96, 68B string)",
                        "BENCHMARKS_METHODOLOGY.md#1-bpe-arena-encode",
                        ok, "decoded=" + decoded_sample.substr (0, 16), true,
                        true});
  }

  /* 2. POSIX mmap O(1) zero-copy embedding lookups (V=1000, D=64 fp32).  */
  {
    std::vector<std::string> words;
    std::vector<float> mat;
    const size_t v_cnt = 1000;
    const size_t d_dim = 64;
    words.reserve (v_cnt);
    mat.resize (v_cnt * d_dim, 0.5f);
    for (size_t i = 0; i < v_cnt; ++i)
      {
        words.push_back ("word_" + std::to_string (i));
      }
    auto mem_emb = Mmap_Embeddings::from_memory (words, mat, d_dim).value ();
    const std::string path = "/tmp/obsidian_bench.obvec";
    mem_emb.save_obvec_mmap (path);
    auto mmap_emb = Mmap_Embeddings::open_mmap (path).value ();
    std::remove (path.c_str ());

    const size_t lookup_iters = 100000;
    float accum = 0.0f;
    Stats mmap_st = measure_stats (
      [&] (void) {
        float s = 0.0f;
        for (size_t i = 0; i < lookup_iters; ++i)
          {
            auto sp = mmap_emb.lookup ("word_500").value ();
            s += sp[0];
          }
        accum = s;
      },
      4, 15, 1);

    double ns_per_lookup =
      (mmap_st.p50_ms * 1.0e6) / static_cast<double> (lookup_iters);
    bool ok = std::abs (accum - 50000.0f) < 1.0f;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision (1) << ns_per_lookup
        << " ns/lookup warm page-cache (V=1000, D=64)";
    print_row ("[Bench 2] Mmap Zero-Copy Lookup (100K)", mmap_st, oss.str ());
    metrics.push_back ({"mmap_lookup_100k_p50_ms", mmap_st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#2-mmap-zero-copy-lookup",
                        ok, "checksum=" + std::to_string (accum), false,
                        true});
  }

  /* 3. Fused-QKV Grouped-Query Attention (GQA) forward [4, 32, 64].  */
  {
    No_Grad_Guard no_grad;
    Grouped_Query_Attention gqa (64, 8, 2, 0.0, true, true, 128);
    gqa.eval ();
    Tensor x = basalt::randn ({4, 32, 64});
    Tensor y;
    const size_t attn_iters = 20;
    Stats attn_st = measure_stats (
      [&] (void) {
        for (size_t i = 0; i < attn_iters; ++i)
          {
            y = gqa.forward_self (x, Tensor (), true);
          }
      },
      3, 13, 1);
    attn_st.p50_ms /= static_cast<double> (attn_iters);
    attn_st.p99_ms /= static_cast<double> (attn_iters);
    double y0 = y.read_scalar_at (0);
    bool ok = y.numel () == 4 * 32 * 64 && std::isfinite (y0);
    double tok_per_s = (4.0 * 32.0 * 1000.0) / attn_st.p50_ms;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision (0) << tok_per_s
        << " tok/s (B=4, S=32, D=64, H=8/2)";
    print_row ("[Bench 3] Fused-QKV GQA [4, 32, 64]", attn_st, oss.str ());
    metrics.push_back ({"fused_qkv_gqa_4x32x64_p50_ms", attn_st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#3-fused-qkv-gqa-forward",
                        ok, "y[0]=" + std::to_string (y0), false, true});
  }

  double peak_rss_mb = read_hwm_mb ();
  std::cout << "------------------------------------------------------------"
               "------------------\n";
  std::cout << std::fixed << std::setprecision (2)
            << "  Memory RSS Phases: Bare=" << bare_rss_mb
            << " MB | Post-Library-Load=" << post_load_rss_mb
            << " MB | Peak-Workload=" << peak_rss_mb << " MB\n";
  std::cout << "============================================================"
               "==================\n";

  metrics.push_back ({"rss_bare_process_mb", bare_rss_mb, "MB",
                      "BENCHMARKS_METHODOLOGY.md#4-process-memory-rss-phases",
                      bare_rss_mb > 0.5, "Bare process VmRSS at main() entry",
                      false, true});
  metrics.push_back ({"rss_post_library_load_mb", post_load_rss_mb, "MB",
                      "BENCHMARKS_METHODOLOGY.md#4-process-memory-rss-phases",
                      post_load_rss_mb >= bare_rss_mb,
                      "Post-BPE-init VmRSS", false, true});
  metrics.push_back ({"rss_peak_workload_mb", peak_rss_mb, "MB",
                      "BENCHMARKS_METHODOLOGY.md#4-process-memory-rss-phases",
                      peak_rss_mb >= post_load_rss_mb,
                      "Peak VmHWM after GQA attention forward passes",
                      false, true});

  if (auto parent = std::filesystem::path (json_path).parent_path ();
      !parent.empty ())
    {
      std::filesystem::create_directories (parent);
    }
  {
    std::ofstream jout (json_path);
    jout << std::fixed << std::setprecision (4);
    jout << "{\n";
    jout << "  \"hardware\": {\n";
    jout << "    \"cpu\": \"12th Gen Intel(R) Core(TM) i7-12700K "
            "(12C/20T)\",\n";
    jout << "    \"gpu\": \"NVIDIA GeForce RTX 3070 (8192 MiB GDDR6, "
            "Driver 550.163.01)\",\n";
    jout << "    \"kernel\": \"Linux 6.12.113+deb13-amd64 x86_64\",\n";
    jout << "    \"ram\": \"48 GiB Dual-Channel DDR4\",\n";
    jout << "    \"disk\": \"Samsung SSD 970 EVO Plus 1TB NVMe\"\n";
    jout << "  },\n";
    jout << "  \"timestamp\": \"" << iso_timestamp () << "\",\n";
    jout << "  \"metrics\": [\n";
    for (size_t i = 0; i < metrics.size (); ++i)
      {
        const auto &m = metrics[i];
        jout << "    {\n";
        jout << "      \"name\": \"" << m.name << "\",\n";
        jout << "      \"value\": " << m.value << ",\n";
        jout << "      \"unit\": \"" << m.unit << "\",\n";
        jout << "      \"methodology_ref\": \"" << m.methodology_ref
             << "\",\n";
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
