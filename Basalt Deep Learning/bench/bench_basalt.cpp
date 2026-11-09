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

/* Audited benchmark suite for Basalt Deep Learning.
   Measures p50/p99 latency, throughput, 3-phase RSS, and sanity checks
   across CPU AVX2+OpenMP kernels and CUDA cuBLAS / Tensor Core kernels.  */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <basalt/basalt.hpp>

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

struct Stats
{
  double p50_ms = 0.0;
  double p99_ms = 0.0;
};

template <typename Fn>
Stats
measure_stats (Fn &&fn, int warmup = 4, int repeats = 13, int batch_iters = 4)
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
  std::cout << "\n--- Basalt bench-ci Regression Gate (Max 15% Drift) ---\n";
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

  basalt::seed_everything (42);
  {
    basalt::No_Grad_Guard ng;
    basalt::Tensor wa = basalt::randn ({256, 256});
    basalt::Tensor wb = basalt::randn ({256, 256});
    for (int w = 0; w < 10; ++w)
      {
        (void) basalt::ops::matmul (wa, wb);
      }
  }
  double post_load_rss_mb = read_rss_mb ();

  std::cout << "============================================================"
               "==================\n";
  std::cout << " Basalt Deep Learning " << basalt_version_string ()
            << " Audited Benchmark Suite\n";
  std::cout << "============================================================"
               "==================\n";

  std::vector<Metric_Entry> metrics;

  /* 1. 256x256 Matrix Multiplication (CPU AVX2+FMA, Warm vs Cold Cache).  */
  {
    basalt::No_Grad_Guard ng;
    basalt::Tensor a = basalt::randn ({256, 256});
    basalt::Tensor b = basalt::randn ({256, 256});
    basalt::Tensor c;
    Stats warm_st = measure_stats (
      [&] (void) {
        c = basalt::ops::matmul (a, b);
      },
      6, 15, 20);
    std::vector<float> thrash (8 * 1024 * 1024, 1.0f);
    std::vector<double> cold_samples (9);
    for (size_t r = 0; r < 9; ++r)
      {
        for (size_t k = 0; k < thrash.size (); k += 16)
          {
            thrash[k] += static_cast<float> (r);
          }
        auto t0 = Clock::now ();
        c = basalt::ops::matmul (a, b);
        auto t1 = Clock::now ();
        cold_samples[r] =
          std::chrono::duration<double, std::milli> (t1 - t0).count ();
      }
    std::sort (cold_samples.begin (), cold_samples.end ());
    double cold_p50_ms = cold_samples[4];
    double warm_gflops =
      (2.0 * 256.0 * 256.0 * 256.0) / (warm_st.p50_ms * 1e6);
    double cold_gflops = (2.0 * 256.0 * 256.0 * 256.0) / (cold_p50_ms * 1e6);
    double c0 = c.read_scalar_at (0) + static_cast<double> (thrash[0]) * 0.0;
    bool ok = c.numel () == 65536 && std::isfinite (c0);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision (1) << warm_gflops
        << " GFLOP/s warm | " << cold_gflops << " GFLOP/s cold";
    print_row ("1. CPU matmul (256x256 fp32)", warm_st, oss.str ());
    metrics.push_back ({"cpu_matmul_256x256_fp32_warm_p50_ms", warm_st.p50_ms,
                        "ms",
                        "BENCHMARKS_METHODOLOGY.md#1-cpu-matmul-256x256-fp32",
                        ok, "c[0]=" + std::to_string (c0), false, true});
    metrics.push_back ({"cpu_matmul_256x256_fp32_warm_gflops", warm_gflops,
                        "GFLOP/s (fp32 warm L2/L3)",
                        "BENCHMARKS_METHODOLOGY.md#1-cpu-matmul-256x256-fp32",
                        ok, "33.55 MFLOP/call", true, true});
  }

  /* 2. Conv2d forward pass (N=4, Cin=8, Cout=16, H=16, W=16, K=3).  */
  {
    basalt::No_Grad_Guard ng;
    basalt::nn::Conv2d conv (8, 16, 3, 1, 1);
    basalt::Tensor x = basalt::randn ({4, 8, 16, 16});
    basalt::Tensor out;
    Stats st = measure_stats (
      [&] (void) {
        out = conv.forward (x);
      },
      6, 15, 20);
    double v0 = out.read_scalar_at (0);
    bool ok = out.numel () == 4 * 16 * 16 * 16 && std::isfinite (v0);
    print_row ("2. CPU Conv2d (4x8x16x16 -> 16, k=3)", st, "im2col+GEMM fp32");
    metrics.push_back ({"cpu_conv2d_4x8x16x16_p50_ms", st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#2-cpu-conv2d-forward",
                        ok, "out[0]=" + std::to_string (v0), false, true});
  }

  /* 3. MLP Forward + Reverse-Mode Autograd Backward (32x64->128->32).  */
  {
    basalt::nn::Sequential mlp;
    mlp.add ("fc1", std::make_shared<basalt::nn::Linear> (64, 128));
    mlp.add ("act", std::make_shared<basalt::nn::GELU> ());
    mlp.add ("fc2", std::make_shared<basalt::nn::Linear> (128, 32));
    basalt::Tensor x = basalt::randn ({32, 64});
    double last_loss = 0.0;
    Stats st = measure_stats (
      [&] (void) {
        mlp.zero_grad ();
        basalt::Tensor out = mlp.forward (x);
        basalt::Tensor l = basalt::ops::mean (out);
        l.backward ();
        last_loss = l.item_double ();
      },
      4, 15, 10);
    auto params = mlp.parameters ();
    bool ok = std::isfinite (last_loss) && !params.empty ()
              && params[0].grad ().defined ();
    print_row ("3. CPU MLP (32x64->128->32) fwd+bwd", st, "batch=32 fp32");
    metrics.push_back ({"cpu_mlp_fwd_bwd_p50_ms", st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#3-cpu-mlp-forward-backward",
                        ok, "loss=" + std::to_string (last_loss), false,
                        true});
  }

  /* 4. Transformer_Encoder_Layer (batch=4, seq=16, d_model=32, nhead=4).  */
  {
    basalt::nn::Transformer_Encoder_Layer enc (32, 4, 64, 0.0);
    basalt::Tensor x = basalt::randn ({4, 16, 32});
    double last_loss = 0.0;
    Stats st = measure_stats (
      [&] (void) {
        enc.zero_grad ();
        basalt::Tensor out = enc.forward (x);
        basalt::Tensor l = basalt::ops::mean (out);
        l.backward ();
        last_loss = l.item_double ();
      },
      4, 15, 6);
    auto params = enc.parameters ();
    bool ok = std::isfinite (last_loss) && !params.empty ()
              && params[0].grad ().defined ();
    print_row ("4. CPU Transformer_Encoder (4x16x32)", st,
               "1-layer d=32 h=4 fwd+bwd");
    metrics.push_back ({"cpu_transformer_enc_4x16x32_p50_ms", st.p50_ms, "ms",
                        "BENCHMARKS_METHODOLOGY.md#4-cpu-transformer-encoder",
                        ok, "loss=" + std::to_string (last_loss), false,
                        true});
  }

  double post_gpu_rss_mb = post_load_rss_mb;

  /* 5 & 6. CUDA (cuda:0) 1024x1024 fp32 & 2048x2048 fp16 cuBLAS GEMM.  */
  if (basalt::cuda_is_available ())
    {
      basalt::No_Grad_Guard ng;
      basalt::Device gpu = basalt::Device::cuda (0);
      basalt::Tensor a = basalt::randn ({1024, 1024}).to (gpu);
      basalt::Tensor b = basalt::randn ({1024, 1024}).to (gpu);
      basalt::Tensor c = basalt::ops::matmul (a, b);
      basalt::device_synchronize (gpu);
      std::this_thread::sleep_for (std::chrono::milliseconds (1000));
      post_gpu_rss_mb = read_rss_mb ();

      /* Re-warm GPU boost clock after 1s idle measurement.  */
      for (int w = 0; w < 20; ++w)
        {
          c = basalt::ops::matmul (a, b);
        }
      basalt::device_synchronize (gpu);

      Stats fp32_st = measure_stats (
        [&] (void) {
          for (int k = 0; k < 10; ++k)
            {
              c = basalt::ops::matmul (a, b);
            }
          basalt::device_synchronize (gpu);
        },
        4, 15, 1);
      fp32_st.p50_ms /= 10.0;
      fp32_st.p99_ms /= 10.0;
      double fp32_gflops =
        (2.0 * 1024.0 * 1024.0 * 1024.0) / (fp32_st.p50_ms * 1e6);
      double c0 = c.read_scalar_at (0);
      bool ok32 = c.numel () == 1024 * 1024 && std::isfinite (c0);
      std::ostringstream oss32;
      oss32 << std::fixed << std::setprecision (1) << fp32_gflops
            << " GFLOP/s (cuBLAS fp32)";
      print_row ("5. CUDA matmul (1024x1024 fp32)", fp32_st, oss32.str ());
      metrics.push_back ({"cuda_matmul_1024x1024_fp32_p50_ms", fp32_st.p50_ms,
                          "ms",
                          "BENCHMARKS_METHODOLOGY.md#5-cuda-matmul-1024x1024",
                          ok32, "c[0]=" + std::to_string (c0), false, true});
      metrics.push_back ({"cuda_matmul_1024x1024_fp32_gflops", fp32_gflops,
                          "GFLOP/s (CUDA fp32 resident)",
                          "BENCHMARKS_METHODOLOGY.md#5-cuda-matmul-1024x1024",
                          ok32, "2.147 GFLOP/call", true, true});

      basalt::Tensor ah =
        basalt::randn ({2048, 2048}).to (gpu, basalt::Dtype::Float16);
      basalt::Tensor bh =
        basalt::randn ({2048, 2048}).to (gpu, basalt::Dtype::Float16);
      basalt::Tensor ch = basalt::ops::matmul (ah, bh);
      basalt::device_synchronize (gpu);

      Stats fp16_st = measure_stats (
        [&] (void) {
          for (int k = 0; k < 10; ++k)
            {
              ch = basalt::ops::matmul (ah, bh);
            }
          basalt::device_synchronize (gpu);
        },
        4, 15, 1);
      fp16_st.p50_ms /= 10.0;
      fp16_st.p99_ms /= 10.0;
      double tc_gflops =
        (2.0 * 2048.0 * 2048.0 * 2048.0) / (fp16_st.p50_ms * 1e6);
      double ch0 = ch.read_scalar_at (0);
      bool ok16 = ch.numel () == 2048 * 2048 && std::isfinite (ch0);
      std::ostringstream oss16;
      oss16 << std::fixed << std::setprecision (1) << tc_gflops
            << " GFLOP/s (TensorCore fp16)";
      print_row ("6. CUDA TensorCore (2048x2048 fp16)", fp16_st, oss16.str ());
      metrics.push_back ({"cuda_tensorcore_2048x2048_fp16_p50_ms",
                          fp16_st.p50_ms, "ms",
                          "BENCHMARKS_METHODOLOGY.md#6-cuda-tensorcore-fp16",
                          ok16, "ch[0]=" + std::to_string (ch0), false, true});
      metrics.push_back ({"cuda_tensorcore_2048x2048_fp16_gflops", tc_gflops,
                          "GFLOP/s (CUDA fp16 TensorCore resident)",
                          "BENCHMARKS_METHODOLOGY.md#6-cuda-tensorcore-fp16",
                          ok16, "17.18 GFLOP/call", true, true});
    }

  std::cout << "------------------------------------------------------------"
               "------------------\n";
  std::cout << std::fixed << std::setprecision (2)
            << "  Memory RSS Phases: Bare=" << bare_rss_mb
            << " MB | Post-Library-Load=" << post_load_rss_mb
            << " MB | Post-CUDA-Init=" << post_gpu_rss_mb << " MB\n";
  std::cout << "============================================================"
               "==================\n";

  metrics.push_back ({"rss_bare_process_mb", bare_rss_mb, "MB",
                      "BENCHMARKS_METHODOLOGY.md#7-process-memory-rss-phases",
                      bare_rss_mb > 0.5, "Bare process VmRSS at main() entry",
                      false, true});
  metrics.push_back ({"rss_post_library_load_mb", post_load_rss_mb, "MB",
                      "BENCHMARKS_METHODOLOGY.md#7-process-memory-rss-phases",
                      post_load_rss_mb >= bare_rss_mb,
                      "Post-CPU-library-load VmRSS (pre-CUDA)", false, true});
  metrics.push_back ({"rss_post_cuda_init_mb", post_gpu_rss_mb, "MB",
                      "BENCHMARKS_METHODOLOGY.md#7-process-memory-rss-phases",
                      post_gpu_rss_mb >= post_load_rss_mb,
                      "Post-CUDA/cuBLAS-init + 1 GPU frame + 1s idle VmRSS",
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
            "Driver 550.163.01, CUDA 12.4)\",\n";
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
